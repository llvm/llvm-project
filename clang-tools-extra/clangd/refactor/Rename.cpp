//===--- Rename.cpp - Symbol-rename refactorings -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Rename.h"
#include "AST.h"
#include "FindTarget.h"
#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <optional>

namespace clang {
namespace clangd {
namespace {

std::optional<std::string> filePath(const SymbolLocation &Loc,
                                    llvm::StringRef HintFilePath) {
  if (!Loc)
    return std::nullopt;
  auto Path = URI::resolve(Loc.FileURI, HintFilePath);
  if (!Path) {
    elog("Could not resolve URI {0}: {1}", Loc.FileURI, Path.takeError());
    return std::nullopt;
  }

  return *Path;
}

// Returns true if the given location is expanded from any macro body.
bool isInMacroBody(const SourceManager &SM, SourceLocation Loc) {
  while (Loc.isMacroID()) {
    if (SM.isMacroBodyExpansion(Loc))
      return true;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return false;
}

// Canonical declarations help simplify the process of renaming. Examples:
// - Template's canonical decl is the templated declaration (i.e.
//   ClassTemplateDecl is canonicalized to its child CXXRecordDecl,
//   FunctionTemplateDecl - to child FunctionDecl)
// - Given a constructor/destructor, canonical declaration is the parent
//   CXXRecordDecl because we want to rename both type name and its ctor/dtor.
// - All specializations are canonicalized to the primary template. For example:
//
//    template <typename T, int U>
//    bool Foo = true; (1)
//
//    template <typename T>
//    bool Foo<T, 0> = true; (2)
//
//    template <>
//    bool Foo<int, 0> = true; (3)
//
// Here, both partial (2) and full (3) specializations are canonicalized to (1)
// which ensures all three of them are renamed.
const NamedDecl *canonicalRenameDecl(const NamedDecl *D) {
  if (const auto *VarTemplate = dyn_cast<VarTemplateSpecializationDecl>(D))
    return canonicalRenameDecl(
        VarTemplate->getSpecializedTemplate()->getTemplatedDecl());
  if (const auto *Template = dyn_cast<TemplateDecl>(D))
    if (const NamedDecl *TemplatedDecl = Template->getTemplatedDecl())
      return canonicalRenameDecl(TemplatedDecl);
  if (const auto *ClassTemplateSpecialization =
          dyn_cast<ClassTemplateSpecializationDecl>(D))
    return canonicalRenameDecl(
        ClassTemplateSpecialization->getSpecializedTemplate()
            ->getTemplatedDecl());
  if (const auto *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (Method->getDeclKind() == Decl::Kind::CXXConstructor ||
        Method->getDeclKind() == Decl::Kind::CXXDestructor)
      return canonicalRenameDecl(Method->getParent());
    if (const FunctionDecl *InstantiatedMethod =
            Method->getInstantiatedFromMemberFunction())
      return canonicalRenameDecl(InstantiatedMethod);
    // FIXME(kirillbobyrev): For virtual methods with
    // size_overridden_methods() > 1, this will not rename all functions it
    // overrides, because this code assumes there is a single canonical
    // declaration.
    if (Method->isVirtual() && Method->size_overridden_methods())
      return canonicalRenameDecl(*Method->overridden_methods().begin());
  }
  if (const auto *Function = dyn_cast<FunctionDecl>(D))
    if (const FunctionTemplateDecl *Template = Function->getPrimaryTemplate())
      return canonicalRenameDecl(Template);
  if (const auto *Field = dyn_cast<FieldDecl>(D)) {
    // This is a hacky way to do something like
    // CXXMethodDecl::getInstantiatedFromMemberFunction for the field because
    // Clang AST does not store relevant information about the field that is
    // instantiated.
    const auto *FieldParent =
        dyn_cast_or_null<CXXRecordDecl>(Field->getParent());
    if (!FieldParent)
      return Field->getCanonicalDecl();
    FieldParent = FieldParent->getTemplateInstantiationPattern();
    // Field is not instantiation.
    if (!FieldParent || Field->getParent() == FieldParent)
      return Field->getCanonicalDecl();
    for (const FieldDecl *Candidate : FieldParent->fields())
      if (Field->getDeclName() == Candidate->getDeclName())
        return Candidate->getCanonicalDecl();
    elog("FieldParent should have field with the same name as Field.");
  }
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (const VarDecl *OriginalVD = VD->getInstantiatedFromStaticDataMember())
      return canonicalRenameDecl(OriginalVD);
  }
  if (const auto *UD = dyn_cast<UsingShadowDecl>(D)) {
    if (const auto *TargetDecl = UD->getTargetDecl())
      return canonicalRenameDecl(TargetDecl);
  }
  return dyn_cast<NamedDecl>(D->getCanonicalDecl());
}

// Some AST nodes can reference multiple declarations. We try to pick the
// relevant one to rename here.
const NamedDecl *pickInterestingTarget(const NamedDecl *D) {
  // We only support renaming the class name, not the category name. This has
  // to be done outside of canonicalization since we don't want a category name
  // reference to be canonicalized to the class.
  if (const auto *CD = dyn_cast<ObjCCategoryDecl>(D))
    if (const auto CI = CD->getClassInterface())
      return CI;
  return D;
}

llvm::DenseSet<const NamedDecl *> locateDeclAt(ParsedAST &AST,
                                               SourceLocation TokenStartLoc) {
  unsigned Offset =
      AST.getSourceManager().getDecomposedSpellingLoc(TokenStartLoc).second;

  SelectionTree Selection = SelectionTree::createRight(
      AST.getASTContext(), AST.getTokens(), Offset, Offset);
  const SelectionTree::Node *SelectedNode = Selection.commonAncestor();
  if (!SelectedNode)
    return {};

  llvm::DenseSet<const NamedDecl *> Result;
  for (const NamedDecl *D :
       targetDecl(SelectedNode->ASTNode,
                  DeclRelation::Alias | DeclRelation::TemplatePattern,
                  AST.getHeuristicResolver())) {
    D = pickInterestingTarget(D);
    Result.insert(canonicalRenameDecl(D));
  }
  return Result;
}

void filterRenameTargets(llvm::DenseSet<const NamedDecl *> &Decls) {
  // For something like
  //     namespace ns { void foo(); }
  //     void bar() { using ns::f^oo; foo(); }
  // locateDeclAt() will return a UsingDecl and foo's actual declaration.
  // For renaming, we're only interested in foo's declaration, so drop the other
  // one. There should never be more than one UsingDecl here, otherwise the
  // rename would be ambiguos anyway.
  auto UD = llvm::find_if(
      Decls, [](const NamedDecl *D) { return llvm::isa<UsingDecl>(D); });
  if (UD != Decls.end()) {
    Decls.erase(UD);
  }
}

// By default, we exclude symbols from system headers and protobuf symbols as
// renaming these symbols would change system/generated files which are unlikely
// to be good candidates for modification.
bool isExcluded(const NamedDecl &RenameDecl) {
  const auto &SM = RenameDecl.getASTContext().getSourceManager();
  return SM.isInSystemHeader(RenameDecl.getLocation()) ||
         isProtoFile(RenameDecl.getLocation(), SM);
}

enum class ReasonToReject {
  NoSymbolFound,
  NoIndexProvided,
  NonIndexable,
  UnsupportedSymbol,
  AmbiguousSymbol,

  // name validation. FIXME: reconcile with InvalidName
  SameName,
};

std::optional<ReasonToReject> renameable(const NamedDecl &RenameDecl,
                                         StringRef MainFilePath,
                                         const SymbolIndex *Index,
                                         const RenameOptions &Opts) {
  trace::Span Tracer("Renameable");
  if (!Opts.RenameVirtual) {
    if (const auto *S = llvm::dyn_cast<CXXMethodDecl>(&RenameDecl)) {
      if (S->isVirtual())
        return ReasonToReject::UnsupportedSymbol;
    }
  }
  // We allow renaming ObjC methods although they don't have a simple
  // identifier.
  const auto *ID = RenameDecl.getIdentifier();
  if (!ID && !isa<ObjCMethodDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  // Filter out symbols that are unsupported in both rename modes.
  if (llvm::isa<NamespaceDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  if (const auto *FD = llvm::dyn_cast<FunctionDecl>(&RenameDecl)) {
    if (FD->isOverloadedOperator())
      return ReasonToReject::UnsupportedSymbol;
  }
  // function-local symbols is safe to rename.
  if (RenameDecl.getParentFunctionOrMethod())
    return std::nullopt;

  if (isExcluded(RenameDecl))
    return ReasonToReject::UnsupportedSymbol;

  // Check whether the symbol being rename is indexable.
  auto &ASTCtx = RenameDecl.getASTContext();
  bool MainFileIsHeader = isHeaderFile(MainFilePath, ASTCtx.getLangOpts());
  bool DeclaredInMainFile =
      isInsideMainFile(RenameDecl.getBeginLoc(), ASTCtx.getSourceManager());
  bool IsMainFileOnly = true;
  if (MainFileIsHeader)
    // main file is a header, the symbol can't be main file only.
    IsMainFileOnly = false;
  else if (!DeclaredInMainFile)
    IsMainFileOnly = false;
  // If the symbol is not indexable, we disallow rename.
  if (!SymbolCollector::shouldCollectSymbol(
          RenameDecl, RenameDecl.getASTContext(), SymbolCollector::Options(),
          IsMainFileOnly))
    return ReasonToReject::NonIndexable;

  return std::nullopt;
}

llvm::Error makeError(ReasonToReject Reason) {
  auto Message = [](ReasonToReject Reason) {
    switch (Reason) {
    case ReasonToReject::NoSymbolFound:
      return "there is no symbol at the given location";
    case ReasonToReject::NoIndexProvided:
      return "no index provided";
    case ReasonToReject::NonIndexable:
      return "symbol may be used in other files (not eligible for indexing)";
    case ReasonToReject::UnsupportedSymbol:
      return "symbol is not a supported kind (e.g. namespace, macro)";
    case ReasonToReject::AmbiguousSymbol:
      return "there are multiple symbols at the given location";
    case ReasonToReject::SameName:
      return "new name is the same as the old name";
    }
    llvm_unreachable("unhandled reason kind");
  };
  return error("Cannot rename symbol: {0}", Message(Reason));
}

// Return all rename occurrences in the main file.
std::vector<SourceLocation> findOccurrencesWithinFile(ParsedAST &AST,
                                                      const NamedDecl &ND) {
  trace::Span Tracer("FindOccurrencesWithinFile");
  assert(canonicalRenameDecl(&ND) == &ND &&
         "ND should be already canonicalized.");

  std::vector<SourceLocation> Results;
  for (Decl *TopLevelDecl : AST.getLocalTopLevelDecls()) {
    findExplicitReferences(
        TopLevelDecl,
        [&](ReferenceLoc Ref) {
          if (Ref.Targets.empty())
            return;
          for (const auto *Target : Ref.Targets) {
            if (canonicalRenameDecl(Target) == &ND) {
              Results.push_back(Ref.NameLoc);
              return;
            }
          }
        },
        AST.getHeuristicResolver());
  }

  return Results;
}

// Detect name conflict with othter DeclStmts in the same enclosing scope.
const NamedDecl *lookupSiblingWithinEnclosingScope(ASTContext &Ctx,
                                                   const NamedDecl &RenamedDecl,
                                                   StringRef NewName) {
  // Store Parents list outside of GetSingleParent, so that returned pointer is
  // not invalidated.
  DynTypedNodeList Storage(DynTypedNode::create(RenamedDecl));
  auto GetSingleParent = [&](const DynTypedNode &Node) -> const DynTypedNode * {
    Storage = Ctx.getParents(Node);
    return (Storage.size() == 1) ? Storage.begin() : nullptr;
  };

  // We need to get to the enclosing scope: NamedDecl's parent is typically
  // DeclStmt (or FunctionProtoTypeLoc in case of function arguments), so
  // enclosing scope would be the second order parent.
  const auto *Parent = GetSingleParent(DynTypedNode::create(RenamedDecl));
  if (!Parent || !(Parent->get<DeclStmt>() || Parent->get<TypeLoc>()))
    return nullptr;
  Parent = GetSingleParent(*Parent);

  // The following helpers check corresponding AST nodes for variable
  // declarations with the name collision.
  auto CheckDeclStmt = [&](const DeclStmt *DS,
                           StringRef Name) -> const NamedDecl * {
    if (!DS)
      return nullptr;
    for (const auto &Child : DS->getDeclGroup())
      if (const auto *ND = dyn_cast<NamedDecl>(Child))
        if (ND != &RenamedDecl && ND->getDeclName().isIdentifier() &&
            ND->getName() == Name &&
            ND->getIdentifierNamespace() & RenamedDecl.getIdentifierNamespace())
          return ND;
    return nullptr;
  };
  auto CheckCompoundStmt = [&](const Stmt *S,
                               StringRef Name) -> const NamedDecl * {
    if (const auto *CS = dyn_cast_or_null<CompoundStmt>(S))
      for (const auto *Node : CS->children())
        if (const auto *Result = CheckDeclStmt(dyn_cast<DeclStmt>(Node), Name))
          return Result;
    return nullptr;
  };
  auto CheckConditionVariable = [&](const auto *Scope,
                                    StringRef Name) -> const NamedDecl * {
    if (!Scope)
      return nullptr;
    return CheckDeclStmt(Scope->getConditionVariableDeclStmt(), Name);
  };

  // CompoundStmt is the most common enclosing scope for function-local symbols
  // In the simplest case we just iterate through sibling DeclStmts and check
  // for collisions.
  if (const auto *EnclosingCS = Parent->get<CompoundStmt>()) {
    if (const auto *Result = CheckCompoundStmt(EnclosingCS, NewName))
      return Result;
    const auto *ScopeParent = GetSingleParent(*Parent);
    // CompoundStmt may be found within if/while/for. In these cases, rename can
    // collide with the init-statement variable decalaration, they should be
    // checked.
    if (const auto *Result =
            CheckConditionVariable(ScopeParent->get<IfStmt>(), NewName))
      return Result;
    if (const auto *Result =
            CheckConditionVariable(ScopeParent->get<WhileStmt>(), NewName))
      return Result;
    if (const auto *For = ScopeParent->get<ForStmt>())
      if (const auto *Result = CheckDeclStmt(
              dyn_cast_or_null<DeclStmt>(For->getInit()), NewName))
        return Result;
    // Also check if there is a name collision with function arguments.
    if (const auto *Function = ScopeParent->get<FunctionDecl>())
      for (const auto *Parameter : Function->parameters())
        if (Parameter->getName() == NewName &&
            Parameter->getIdentifierNamespace() &
                RenamedDecl.getIdentifierNamespace())
          return Parameter;
    return nullptr;
  }

  // When renaming a variable within init-statement within if/while/for
  // condition, also check the CompoundStmt in the body.
  if (const auto *EnclosingIf = Parent->get<IfStmt>()) {
    if (const auto *Result = CheckCompoundStmt(EnclosingIf->getElse(), NewName))
      return Result;
    return CheckCompoundStmt(EnclosingIf->getThen(), NewName);
  }
  if (const auto *EnclosingWhile = Parent->get<WhileStmt>())
    return CheckCompoundStmt(EnclosingWhile->getBody(), NewName);
  if (const auto *EnclosingFor = Parent->get<ForStmt>()) {
    // Check for conflicts with other declarations within initialization
    // statement.
    if (const auto *Result = CheckDeclStmt(
            dyn_cast_or_null<DeclStmt>(EnclosingFor->getInit()), NewName))
      return Result;
    return CheckCompoundStmt(EnclosingFor->getBody(), NewName);
  }
  if (const auto *EnclosingFunction = Parent->get<FunctionDecl>()) {
    // Check for conflicts with other arguments.
    for (const auto *Parameter : EnclosingFunction->parameters())
      if (Parameter != &RenamedDecl && Parameter->getName() == NewName &&
          Parameter->getIdentifierNamespace() &
              RenamedDecl.getIdentifierNamespace())
        return Parameter;
    // FIXME: We don't modify all references to function parameters when
    // renaming from forward declaration now, so using a name colliding with
    // something in the definition's body is a valid transformation.
    if (!EnclosingFunction->doesThisDeclarationHaveABody())
      return nullptr;
    return CheckCompoundStmt(EnclosingFunction->getBody(), NewName);
  }

  return nullptr;
}

// Lookup the declarations (if any) with the given Name in the context of
// RenameDecl.
const NamedDecl *lookupSiblingsWithinContext(ASTContext &Ctx,
                                             const NamedDecl &RenamedDecl,
                                             llvm::StringRef NewName) {
  const auto &II = Ctx.Idents.get(NewName);
  DeclarationName LookupName(&II);
  DeclContextLookupResult LookupResult;
  const auto *DC = RenamedDecl.getDeclContext();
  while (DC->isTransparentContext())
    DC = DC->getParent();
  switch (DC->getDeclKind()) {
  // The enclosing DeclContext may not be the enclosing scope, it might have
  // false positives and negatives, so we only choose "confident" DeclContexts
  // that don't have any subscopes that are neither DeclContexts nor
  // transparent.
  //
  // Notably, FunctionDecl is excluded -- because local variables are not scoped
  // to the function, but rather to the CompoundStmt that is its body. Lookup
  // will not find function-local variables.
  case Decl::TranslationUnit:
  case Decl::Namespace:
  case Decl::Record:
  case Decl::Enum:
  case Decl::CXXRecord:
    LookupResult = DC->lookup(LookupName);
    break;
  default:
    break;
  }
  // Lookup may contain the RenameDecl itself, exclude it.
  for (const auto *D : LookupResult)
    if (D->getCanonicalDecl() != RenamedDecl.getCanonicalDecl() &&
        D->getIdentifierNamespace() & RenamedDecl.getIdentifierNamespace())
      return D;
  return nullptr;
}

const NamedDecl *lookupSiblingWithName(ASTContext &Ctx,
                                       const NamedDecl &RenamedDecl,
                                       llvm::StringRef NewName) {
  trace::Span Tracer("LookupSiblingWithName");
  if (const auto *Result =
          lookupSiblingsWithinContext(Ctx, RenamedDecl, NewName))
    return Result;
  return lookupSiblingWithinEnclosingScope(Ctx, RenamedDecl, NewName);
}

struct InvalidName {
  enum Kind {
    Keywords,
    Conflict,
    BadIdentifier,
  };
  Kind K;
  std::string Details;
};
std::string toString(InvalidName::Kind K) {
  switch (K) {
  case InvalidName::Keywords:
    return "Keywords";
  case InvalidName::Conflict:
    return "Conflict";
  case InvalidName::BadIdentifier:
    return "BadIdentifier";
  }
  llvm_unreachable("unhandled InvalidName kind");
}

llvm::Error makeError(InvalidName Reason) {
  auto Message = [](const InvalidName &Reason) {
    switch (Reason.K) {
    case InvalidName::Keywords:
      return llvm::formatv("the chosen name \"{0}\" is a keyword",
                           Reason.Details);
    case InvalidName::Conflict:
      return llvm::formatv("conflict with the symbol in {0}", Reason.Details);
    case InvalidName::BadIdentifier:
      return llvm::formatv("the chosen name \"{0}\" is not a valid identifier",
                           Reason.Details);
    }
    llvm_unreachable("unhandled InvalidName kind");
  };
  return error("invalid name: {0}", Message(Reason));
}

static bool mayBeValidIdentifier(llvm::StringRef Ident, bool AllowColon) {
  assert(llvm::json::isUTF8(Ident));
  if (Ident.empty())
    return false;
  // We don't check all the rules for non-ascii characters (most are allowed).
  bool AllowDollar = true; // lenient
  if (llvm::isASCII(Ident.front()) &&
      !isAsciiIdentifierStart(Ident.front(), AllowDollar))
    return false;
  for (char C : Ident) {
    if (AllowColon && C == ':')
      continue;
    if (llvm::isASCII(C) && !isAsciiIdentifierContinue(C, AllowDollar))
      return false;
  }
  return true;
}

std::string getName(const NamedDecl &RenameDecl) {
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(&RenameDecl))
    return MD->getSelector().getAsString();
  if (const auto *ID = RenameDecl.getIdentifier())
    return ID->getName().str();
  return "";
}

// Check if we can rename the given RenameDecl into NewName.
// Return details if the rename would produce a conflict.
llvm::Error checkName(const NamedDecl &RenameDecl, llvm::StringRef NewName,
                      llvm::StringRef OldName) {
  trace::Span Tracer("CheckName");
  static constexpr trace::Metric InvalidNameMetric(
      "rename_name_invalid", trace::Metric::Counter, "invalid_kind");

  if (OldName == NewName)
    return makeError(ReasonToReject::SameName);

  if (const auto *MD = dyn_cast<ObjCMethodDecl>(&RenameDecl)) {
    const auto Sel = MD->getSelector();
    if (Sel.getNumArgs() != NewName.count(':') &&
        NewName != "__clangd_rename_placeholder")
      return makeError(InvalidName{InvalidName::BadIdentifier, NewName.str()});
  }

  auto &ASTCtx = RenameDecl.getASTContext();
  std::optional<InvalidName> Result;
  if (isKeyword(NewName, ASTCtx.getLangOpts()))
    Result = InvalidName{InvalidName::Keywords, NewName.str()};
  else if (!mayBeValidIdentifier(NewName, isa<ObjCMethodDecl>(&RenameDecl)))
    Result = InvalidName{InvalidName::BadIdentifier, NewName.str()};
  else {
    // Name conflict detection.
    // Function conflicts are subtle (overloading), so ignore them.
    if (RenameDecl.getKind() != Decl::Function &&
        RenameDecl.getKind() != Decl::CXXMethod) {
      if (auto *Conflict = lookupSiblingWithName(ASTCtx, RenameDecl, NewName))
        Result = InvalidName{
            InvalidName::Conflict,
            Conflict->getLocation().printToString(ASTCtx.getSourceManager())};
    }
  }
  if (Result) {
    InvalidNameMetric.record(1, toString(Result->K));
    return makeError(*Result);
  }
  return llvm::Error::success();
}

bool isSelectorLike(const syntax::Token &Cur, const syntax::Token &Next) {
  return Cur.kind() == tok::identifier && Next.kind() == tok::colon &&
         // We require the selector name and : to be contiguous.
         // e.g. support `foo:` but not `foo :`.
         Cur.endLocation() == Next.location();
}

bool isMatchingSelectorName(const syntax::Token &Cur, const syntax::Token &Next,
                            const SourceManager &SM,
                            llvm::StringRef SelectorName) {
  if (SelectorName.empty())
    return Cur.kind() == tok::colon;
  return isSelectorLike(Cur, Next) && Cur.text(SM) == SelectorName;
}

// Scan through Tokens to find ranges for each selector fragment in Sel assuming
// its first segment is located at Tokens.front().
// The search will terminate upon seeing Terminator or a ; at the top level.
std::optional<SymbolRange>
findAllSelectorPieces(llvm::ArrayRef<syntax::Token> Tokens,
                      const SourceManager &SM, const RenameSymbolName &Name,
                      tok::TokenKind Terminator) {
  assert(!Tokens.empty());

  ArrayRef<std::string> NamePieces = Name.getNamePieces();
  unsigned NumArgs = NamePieces.size();
  llvm::SmallVector<tok::TokenKind, 8> Closes;
  std::vector<Range> SelectorPieces;
  for (unsigned Index = 0, Last = Tokens.size(); Index < Last - 1; ++Index) {
    const auto &Tok = Tokens[Index];

    if (Closes.empty()) {
      auto PieceCount = SelectorPieces.size();
      if (PieceCount < NumArgs &&
          isMatchingSelectorName(Tok, Tokens[Index + 1], SM,
                                 NamePieces[PieceCount])) {
        // If 'foo:' instead of ':' (empty selector), we need to skip the ':'
        // token after the name. We don't currently properly support empty
        // selectors since we may lex them improperly due to ternary statements
        // as well as don't properly support storing their ranges for edits.
        if (!NamePieces[PieceCount].empty())
          ++Index;
        SelectorPieces.push_back(
            halfOpenToRange(SM, Tok.range(SM).toCharRange(SM)));
        continue;
      }
      // If we've found all pieces but the current token looks like another
      // selector piece, it means the method being renamed is a strict prefix of
      // the selector we've found - should be skipped.
      if (SelectorPieces.size() >= NumArgs &&
          isSelectorLike(Tok, Tokens[Index + 1]))
        return std::nullopt;
    }

    if (Closes.empty() && Tok.kind() == Terminator)
      return SelectorPieces.size() == NumArgs
                 ? std::optional(SymbolRange(SelectorPieces))
                 : std::nullopt;

    switch (Tok.kind()) {
    case tok::l_square:
      Closes.push_back(tok::r_square);
      break;
    case tok::l_paren:
      Closes.push_back(tok::r_paren);
      break;
    case tok::l_brace:
      Closes.push_back(tok::r_brace);
      break;
    case tok::r_square:
    case tok::r_paren:
    case tok::r_brace:
      if (Closes.empty() || Closes.back() != Tok.kind())
        return std::nullopt;
      Closes.pop_back();
      break;
    case tok::semi:
      // top level ; terminates all statements.
      if (Closes.empty())
        return SelectorPieces.size() == NumArgs
                   ? std::optional(SymbolRange(SelectorPieces))
                   : std::nullopt;
      break;
    default:
      break;
    }
  }
  return std::nullopt;
}

/// Collects all ranges of the given identifier/selector in the source code.
///
/// If `Name` is an Objective-C symbol name, this does a full lex of the given
/// source code in order to identify all selector fragments (e.g. in method
/// exprs/decls) since they are non-contiguous.
std::vector<SymbolRange>
collectRenameIdentifierRanges(const RenameSymbolName &Name,
                              llvm::StringRef Content,
                              const LangOptions &LangOpts) {
  std::vector<SymbolRange> Ranges;
  if (auto SinglePiece = Name.getSinglePiece()) {
    auto IdentifierRanges =
        collectIdentifierRanges(*SinglePiece, Content, LangOpts);
    for (const auto &R : IdentifierRanges)
      Ranges.emplace_back(R);
    return Ranges;
  }
  // FIXME: InMemoryFileAdapter crashes unless the buffer is null terminated!
  std::string NullTerminatedCode = Content.str();
  SourceManagerForFile FileSM("mock_file_name.cpp", NullTerminatedCode);
  auto &SM = FileSM.get();

  // We track parens and brackets to ensure that we don't accidentally try
  // parsing a method declaration or definition which isn't at the top level or
  // similar looking expressions (e.g. an @selector() expression).
  llvm::SmallVector<tok::TokenKind, 8> Closes;
  llvm::StringRef FirstSelPiece = Name.getNamePieces()[0];

  auto Tokens = syntax::tokenize(SM.getMainFileID(), SM, LangOpts);
  unsigned Last = Tokens.size() - 1;
  for (unsigned Index = 0; Index < Last; ++Index) {
    const auto &Tok = Tokens[Index];

    // Search for the first selector piece to begin a match, but make sure we're
    // not in () to avoid the @selector(foo:bar:) case.
    if ((Closes.empty() || Closes.back() == tok::r_square) &&
        isMatchingSelectorName(Tok, Tokens[Index + 1], SM, FirstSelPiece)) {
      // We found a candidate for our match, this might be a method call,
      // declaration, or unrelated identifier eg:
      // - [obj ^sel0: X sel1: Y ... ]
      //
      // or
      //
      // @interface Foo
      //  - (int)^sel0:(int)x sel1:(int)y;
      // @end
      //
      // or
      //
      // @implementation Foo
      //  - (int)^sel0:(int)x sel1:(int)y {}
      // @end
      //
      // but not @selector(sel0:sel1:)
      //
      // Check if we can find all the relevant selector peices starting from
      // this token
      auto SelectorRanges =
          findAllSelectorPieces(ArrayRef(Tokens).slice(Index), SM, Name,
                                Closes.empty() ? tok::l_brace : Closes.back());
      if (SelectorRanges)
        Ranges.emplace_back(std::move(*SelectorRanges));
    }

    switch (Tok.kind()) {
    case tok::l_square:
      Closes.push_back(tok::r_square);
      break;
    case tok::l_paren:
      Closes.push_back(tok::r_paren);
      break;
    case tok::r_square:
    case tok::r_paren:
      if (Closes.empty()) // Invalid code, give up on the rename.
        return std::vector<SymbolRange>();

      if (Closes.back() == Tok.kind())
        Closes.pop_back();
      break;
    default:
      break;
    }
  }
  return Ranges;
}

clangd::Range tokenRangeForLoc(ParsedAST &AST, SourceLocation TokLoc,
                               const SourceManager &SM,
                               const LangOptions &LangOpts) {
  const auto *Token = AST.getTokens().spelledTokenContaining(TokLoc);
  assert(Token && "rename expects spelled tokens");
  clangd::Range Result;
  Result.start = sourceLocToPosition(SM, Token->location());
  Result.end = sourceLocToPosition(SM, Token->endLocation());
  return Result;
}

// AST-based ObjC method rename, it renames all occurrences in the main file
// even for selectors which may have multiple tokens.
llvm::Expected<tooling::Replacements>
renameObjCMethodWithinFile(ParsedAST &AST, const ObjCMethodDecl *MD,
                           llvm::StringRef NewName,
                           std::vector<SourceLocation> SelectorOccurences) {
  const SourceManager &SM = AST.getSourceManager();
  auto Code = SM.getBufferData(SM.getMainFileID());
  llvm::SmallVector<llvm::StringRef, 8> NewNames;
  NewName.split(NewNames, ":");

  std::vector<Range> Ranges;
  const auto &LangOpts = MD->getASTContext().getLangOpts();
  for (const auto &Loc : SelectorOccurences)
    Ranges.push_back(tokenRangeForLoc(AST, Loc, SM, LangOpts));
  auto FilePath = AST.tuPath();
  auto RenameRanges = collectRenameIdentifierRanges(
      RenameSymbolName(MD->getDeclName()), Code, LangOpts);
  auto RenameEdit = buildRenameEdit(FilePath, Code, RenameRanges, NewNames);
  if (!RenameEdit)
    return error("failed to rename in file {0}: {1}", FilePath,
                 RenameEdit.takeError());
  return RenameEdit->Replacements;
}

// AST-based rename, it renames all occurrences in the main file.
llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, const NamedDecl &RenameDecl,
                 llvm::StringRef NewName) {
  trace::Span Tracer("RenameWithinFile");
  const SourceManager &SM = AST.getSourceManager();

  tooling::Replacements FilteredChanges;
  std::vector<SourceLocation> Locs;
  for (SourceLocation Loc : findOccurrencesWithinFile(AST, RenameDecl)) {
    SourceLocation RenameLoc = Loc;
    // We don't rename in any macro bodies, but we allow rename the symbol
    // spelled in a top-level macro argument in the main file.
    if (RenameLoc.isMacroID()) {
      if (isInMacroBody(SM, RenameLoc))
        continue;
      RenameLoc = SM.getSpellingLoc(Loc);
    }
    // Filter out locations not from main file.
    // We traverse only main file decls, but locations could come from an
    // non-preamble #include file e.g.
    //   void test() {
    //     int f^oo;
    //     #include "use_foo.inc"
    //   }
    if (!isInsideMainFile(RenameLoc, SM))
      continue;
    Locs.push_back(RenameLoc);
  }
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(&RenameDecl)) {
    // The custom ObjC selector logic doesn't handle the zero arg selector
    // case, as it relies on parsing selectors via the trailing `:`.
    // We also choose to use regular rename logic for the single-arg selectors
    // as the AST/Index has the right locations in that case.
    if (MD->getSelector().getNumArgs() > 1)
      return renameObjCMethodWithinFile(AST, MD, NewName, std::move(Locs));

    // Eat trailing : for single argument methods since they're actually
    // considered a separate token during rename.
    NewName.consume_back(":");
  }
  for (const auto &Loc : Locs) {
    if (auto Err = FilteredChanges.add(tooling::Replacement(
            SM, CharSourceRange::getTokenRange(Loc), NewName)))
      return std::move(Err);
  }
  return FilteredChanges;
}

Range toRange(const SymbolLocation &L) {
  Range R;
  R.start.line = L.Start.line();
  R.start.character = L.Start.column();
  R.end.line = L.End.line();
  R.end.character = L.End.column();
  return R;
}

// Walk down from a virtual method to overriding methods, we rename them as a
// group. Note that canonicalRenameDecl() ensures we're starting from the base
// method.
void insertTransitiveOverrides(SymbolID Base, llvm::DenseSet<SymbolID> &IDs,
                               const SymbolIndex &Index) {
  RelationsRequest Req;
  Req.Predicate = RelationKind::OverriddenBy;

  llvm::DenseSet<SymbolID> Pending = {Base};
  while (!Pending.empty()) {
    Req.Subjects = std::move(Pending);
    Pending.clear();

    Index.relations(Req, [&](const SymbolID &, const Symbol &Override) {
      if (IDs.insert(Override.ID).second)
        Pending.insert(Override.ID);
    });
  }
}

// Return all rename occurrences (using the index) outside of the main file,
// grouped by the absolute file path.
llvm::Expected<llvm::StringMap<std::vector<Range>>>
findOccurrencesOutsideFile(const NamedDecl &RenameDecl,
                           llvm::StringRef MainFile, const SymbolIndex &Index,
                           size_t MaxLimitFiles) {
  trace::Span Tracer("FindOccurrencesOutsideFile");
  RefsRequest RQuest;
  RQuest.IDs.insert(getSymbolID(&RenameDecl));

  if (const auto *MethodDecl = llvm::dyn_cast<CXXMethodDecl>(&RenameDecl))
    if (MethodDecl->isVirtual())
      insertTransitiveOverrides(*RQuest.IDs.begin(), RQuest.IDs, Index);

  // Absolute file path => rename occurrences in that file.
  llvm::StringMap<std::vector<Range>> AffectedFiles;
  bool HasMore = Index.refs(RQuest, [&](const Ref &R) {
    if (AffectedFiles.size() >= MaxLimitFiles)
      return;
    if ((R.Kind & RefKind::Spelled) == RefKind::Unknown)
      return;
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (!pathEqual(*RefFilePath, MainFile))
        AffectedFiles[*RefFilePath].push_back(toRange(R.Location));
    }
  });

  if (AffectedFiles.size() >= MaxLimitFiles)
    return error("The number of affected files exceeds the max limit {0}",
                 MaxLimitFiles);
  if (HasMore)
    return error("The symbol {0} has too many occurrences",
                 RenameDecl.getQualifiedNameAsString());
  // Sort and deduplicate the results, in case that index returns duplications.
  for (auto &FileAndOccurrences : AffectedFiles) {
    auto &Ranges = FileAndOccurrences.getValue();
    llvm::sort(Ranges);
    Ranges.erase(llvm::unique(Ranges), Ranges.end());

    SPAN_ATTACH(Tracer, FileAndOccurrences.first(),
                static_cast<int64_t>(Ranges.size()));
  }
  return AffectedFiles;
}

// Index-based rename, it renames all occurrences outside of the main file.
//
// The cross-file rename is purely based on the index, as we don't want to
// build all ASTs for affected files, which may cause a performance hit.
// We choose to trade off some correctness for performance and scalability.
//
// Clangd builds a dynamic index for all opened files on top of the static
// index of the whole codebase. Dynamic index is up-to-date (respects dirty
// buffers) as long as clangd finishes processing opened files, while static
// index (background index) is relatively stale. We choose the dirty buffers
// as the file content we rename on, and fallback to file content on disk if
// there is no dirty buffer.
llvm::Expected<FileEdits>
renameOutsideFile(const NamedDecl &RenameDecl, llvm::StringRef MainFilePath,
                  llvm::StringRef NewName, const SymbolIndex &Index,
                  size_t MaxLimitFiles, llvm::vfs::FileSystem &FS) {
  trace::Span Tracer("RenameOutsideFile");
  auto AffectedFiles = findOccurrencesOutsideFile(RenameDecl, MainFilePath,
                                                  Index, MaxLimitFiles);
  if (!AffectedFiles)
    return AffectedFiles.takeError();
  FileEdits Results;
  for (auto &FileAndOccurrences : *AffectedFiles) {
    llvm::StringRef FilePath = FileAndOccurrences.first();

    auto ExpBuffer = FS.getBufferForFile(FilePath);
    if (!ExpBuffer) {
      elog("Fail to read file content: Fail to open file {0}: {1}", FilePath,
           ExpBuffer.getError().message());
      continue;
    }
    RenameSymbolName RenameName(RenameDecl.getDeclName());
    llvm::SmallVector<llvm::StringRef, 8> NewNames;
    NewName.split(NewNames, ":");

    auto AffectedFileCode = (*ExpBuffer)->getBuffer();
    auto RenameRanges = adjustRenameRanges(
        AffectedFileCode, RenameName, std::move(FileAndOccurrences.second),
        RenameDecl.getASTContext().getLangOpts());
    if (!RenameRanges) {
      // Our heuristics fails to adjust rename ranges to the current state of
      // the file, it is most likely the index is stale, so we give up the
      // entire rename.
      return error("Index results don't match the content of file {0} "
                   "(the index may be stale)",
                   FilePath);
    }
    auto RenameEdit =
        buildRenameEdit(FilePath, AffectedFileCode, *RenameRanges, NewNames);
    if (!RenameEdit)
      return error("failed to rename in file {0}: {1}", FilePath,
                   RenameEdit.takeError());
    if (!RenameEdit->Replacements.empty())
      Results.insert({FilePath, std::move(*RenameEdit)});
  }
  return Results;
}

// A simple edit is either changing line or column, but not both.
bool impliesSimpleEdit(const Position &LHS, const Position &RHS) {
  return LHS.line == RHS.line || LHS.character == RHS.character;
}

// Performs a DFS to enumerate all possible near-miss matches.
// It finds the locations where the indexed occurrences are now spelled in
// Lexed occurrences, a near miss is defined as:
//   - a near miss maps all of the **name** occurrences from the index onto a
//     *subset* of lexed occurrences (we allow a single name refers to more
//     than one symbol)
//   - all indexed occurrences must be mapped, and Result must be distinct and
//     preserve order (only support detecting simple edits to ensure a
//     robust mapping)
//   - each indexed -> lexed occurrences mapping correspondence may change the
//     *line* or *column*, but not both (increases chance of a robust mapping)
void findNearMiss(
    std::vector<size_t> &PartialMatch, ArrayRef<Range> IndexedRest,
    ArrayRef<SymbolRange> LexedRest, int LexedIndex, int &Fuel,
    llvm::function_ref<void(const std::vector<size_t> &)> MatchedCB) {
  if (--Fuel < 0)
    return;
  if (IndexedRest.size() > LexedRest.size())
    return;
  if (IndexedRest.empty()) {
    MatchedCB(PartialMatch);
    return;
  }
  if (impliesSimpleEdit(IndexedRest.front().start,
                        LexedRest.front().range().start)) {
    PartialMatch.push_back(LexedIndex);
    findNearMiss(PartialMatch, IndexedRest.drop_front(), LexedRest.drop_front(),
                 LexedIndex + 1, Fuel, MatchedCB);
    PartialMatch.pop_back();
  }
  findNearMiss(PartialMatch, IndexedRest, LexedRest.drop_front(),
               LexedIndex + 1, Fuel, MatchedCB);
}

} // namespace

RenameSymbolName::RenameSymbolName() : NamePieces({}) {}

namespace {
std::vector<std::string> extractNamePieces(const DeclarationName &DeclName) {
  if (DeclName.getNameKind() ==
          DeclarationName::NameKind::ObjCMultiArgSelector ||
      DeclName.getNameKind() == DeclarationName::NameKind::ObjCOneArgSelector) {
    const Selector &Sel = DeclName.getObjCSelector();
    std::vector<std::string> Result;
    for (unsigned Slot = 0; Slot < Sel.getNumArgs(); ++Slot) {
      Result.push_back(Sel.getNameForSlot(Slot).str());
    }
    return Result;
  }
  return {DeclName.getAsString()};
}
} // namespace

RenameSymbolName::RenameSymbolName(const DeclarationName &DeclName)
    : RenameSymbolName(extractNamePieces(DeclName)) {}

RenameSymbolName::RenameSymbolName(ArrayRef<std::string> NamePieces) {
  for (const auto &Piece : NamePieces)
    this->NamePieces.push_back(Piece);
}

std::optional<std::string> RenameSymbolName::getSinglePiece() const {
  if (getNamePieces().size() == 1) {
    return NamePieces.front();
  }
  return std::nullopt;
}

std::string RenameSymbolName::getAsString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  this->print(OS);
  return Result;
}

void RenameSymbolName::print(raw_ostream &OS) const {
  llvm::interleave(NamePieces, OS, ":");
}

SymbolRange::SymbolRange(Range R) : Ranges({R}) {}

SymbolRange::SymbolRange(std::vector<Range> Ranges)
    : Ranges(std::move(Ranges)) {}

Range SymbolRange::range() const { return Ranges.front(); }

bool operator==(const SymbolRange &LHS, const SymbolRange &RHS) {
  return LHS.Ranges == RHS.Ranges;
}
bool operator!=(const SymbolRange &LHS, const SymbolRange &RHS) {
  return !(LHS == RHS);
}
bool operator<(const SymbolRange &LHS, const SymbolRange &RHS) {
  return LHS.range() < RHS.range();
}

llvm::Expected<RenameResult> rename(const RenameInputs &RInputs) {
  assert(!RInputs.Index == !RInputs.FS &&
         "Index and FS must either both be specified or both null.");
  trace::Span Tracer("Rename flow");
  const auto &Opts = RInputs.Opts;
  ParsedAST &AST = RInputs.AST;
  const SourceManager &SM = AST.getSourceManager();
  llvm::StringRef MainFileCode = SM.getBufferData(SM.getMainFileID());
  // Try to find the tokens adjacent to the cursor position.
  auto Loc = sourceLocationInMainFile(SM, RInputs.Pos);
  if (!Loc)
    return Loc.takeError();
  const syntax::Token *IdentifierToken =
      spelledIdentifierTouching(*Loc, AST.getTokens());

  // Renames should only triggered on identifiers.
  if (!IdentifierToken)
    return makeError(ReasonToReject::NoSymbolFound);
  Range CurrentIdentifier = halfOpenToRange(
      SM, CharSourceRange::getCharRange(IdentifierToken->location(),
                                        IdentifierToken->endLocation()));
  // FIXME: Renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(*IdentifierToken, AST.getPreprocessor()))
    return makeError(ReasonToReject::UnsupportedSymbol);

  auto DeclsUnderCursor = locateDeclAt(AST, IdentifierToken->location());
  filterRenameTargets(DeclsUnderCursor);
  if (DeclsUnderCursor.empty())
    return makeError(ReasonToReject::NoSymbolFound);
  if (DeclsUnderCursor.size() > 1)
    return makeError(ReasonToReject::AmbiguousSymbol);

  const auto &RenameDecl = **DeclsUnderCursor.begin();
  static constexpr trace::Metric RenameTriggerCounter(
      "rename_trigger_count", trace::Metric::Counter, "decl_kind");
  RenameTriggerCounter.record(1, RenameDecl.getDeclKindName());

  std::string Placeholder = getName(RenameDecl);
  auto Invalid = checkName(RenameDecl, RInputs.NewName, Placeholder);
  if (Invalid)
    return std::move(Invalid);

  auto Reject =
      renameable(RenameDecl, RInputs.MainFilePath, RInputs.Index, Opts);
  if (Reject)
    return makeError(*Reject);

  // We have two implementations of the rename:
  //   - AST-based rename: used for renaming local symbols, e.g. variables
  //     defined in a function body;
  //   - index-based rename: used for renaming non-local symbols, and not
  //     feasible for local symbols (as by design our index don't index these
  //     symbols by design;
  // To make cross-file rename work for local symbol, we use a hybrid solution:
  //   - run AST-based rename on the main file;
  //   - run index-based rename on other affected files;
  auto MainFileRenameEdit = renameWithinFile(AST, RenameDecl, RInputs.NewName);
  if (!MainFileRenameEdit)
    return MainFileRenameEdit.takeError();

  llvm::DenseSet<Range> RenamedRanges;
  if (!isa<ObjCMethodDecl>(RenameDecl)) {
    // TODO: Insert the ranges from the ObjCMethodDecl/ObjCMessageExpr selector
    // pieces which are being renamed. This will require us to make changes to
    // locateDeclAt to preserve this AST node.
    RenamedRanges.insert(CurrentIdentifier);
  }

  // Check the rename-triggering location is actually being renamed.
  // This is a robustness check to avoid surprising rename results -- if the
  // the triggering location is not actually the name of the node we identified
  // (e.g. for broken code), then rename is likely not what users expect, so we
  // reject this kind of rename.
  for (const auto &Range : RenamedRanges) {
    auto StartOffset = positionToOffset(MainFileCode, Range.start);
    auto EndOffset = positionToOffset(MainFileCode, Range.end);
    if (!StartOffset)
      return StartOffset.takeError();
    if (!EndOffset)
      return EndOffset.takeError();
    if (llvm::none_of(
            *MainFileRenameEdit,
            [&StartOffset, &EndOffset](const clang::tooling::Replacement &R) {
              return R.getOffset() == *StartOffset &&
                     R.getLength() == *EndOffset - *StartOffset;
            })) {
      return makeError(ReasonToReject::NoSymbolFound);
    }
  }
  RenameResult Result;
  Result.Target = CurrentIdentifier;
  Result.Placeholder = Placeholder;
  Edit MainFileEdits = Edit(MainFileCode, std::move(*MainFileRenameEdit));
  for (const TextEdit &TE : MainFileEdits.asTextEdits())
    Result.LocalChanges.push_back(TE.range);

  // return the main file edit if this is a within-file rename or the symbol
  // being renamed is function local.
  if (RenameDecl.getParentFunctionOrMethod()) {
    Result.GlobalChanges = FileEdits(
        {std::make_pair(RInputs.MainFilePath, std::move(MainFileEdits))});
    return Result;
  }

  // If the index is nullptr, we don't know the completeness of the result, so
  // we don't populate the field GlobalChanges.
  if (!RInputs.Index) {
    assert(Result.GlobalChanges.empty());
    return Result;
  }

  auto OtherFilesEdits = renameOutsideFile(
      RenameDecl, RInputs.MainFilePath, RInputs.NewName, *RInputs.Index,
      Opts.LimitFiles == 0 ? std::numeric_limits<size_t>::max()
                           : Opts.LimitFiles,
      *RInputs.FS);
  if (!OtherFilesEdits)
    return OtherFilesEdits.takeError();
  Result.GlobalChanges = *OtherFilesEdits;
  // Attach the rename edits for the main file.
  Result.GlobalChanges.try_emplace(RInputs.MainFilePath,
                                   std::move(MainFileEdits));
  return Result;
}

llvm::Expected<Edit> buildRenameEdit(llvm::StringRef AbsFilePath,
                                     llvm::StringRef InitialCode,
                                     std::vector<SymbolRange> Occurrences,
                                     llvm::ArrayRef<llvm::StringRef> NewNames) {
  trace::Span Tracer("BuildRenameEdit");
  SPAN_ATTACH(Tracer, "file_path", AbsFilePath);
  SPAN_ATTACH(Tracer, "rename_occurrences",
              static_cast<int64_t>(Occurrences.size()));

  assert(llvm::is_sorted(Occurrences));
  assert(llvm::unique(Occurrences) == Occurrences.end() &&
         "Occurrences must be unique");

  // These two always correspond to the same position.
  Position LastPos{0, 0};
  size_t LastOffset = 0;

  auto Offset = [&](const Position &P) -> llvm::Expected<size_t> {
    assert(LastPos <= P && "malformed input");
    Position Shifted = {
        P.line - LastPos.line,
        P.line > LastPos.line ? P.character : P.character - LastPos.character};
    auto ShiftedOffset =
        positionToOffset(InitialCode.substr(LastOffset), Shifted);
    if (!ShiftedOffset)
      return error("fail to convert the position {0} to offset ({1})", P,
                   ShiftedOffset.takeError());
    LastPos = P;
    LastOffset += *ShiftedOffset;
    return LastOffset;
  };

  struct OccurrenceOffset {
    size_t Start;
    size_t End;
    llvm::StringRef NewName;

    OccurrenceOffset(size_t Start, size_t End, llvm::StringRef NewName)
        : Start(Start), End(End), NewName(NewName) {}
  };

  std::vector<OccurrenceOffset> OccurrencesOffsets;
  for (const auto &SR : Occurrences) {
    for (auto [Range, NewName] : llvm::zip(SR.Ranges, NewNames)) {
      auto StartOffset = Offset(Range.start);
      if (!StartOffset)
        return StartOffset.takeError();
      auto EndOffset = Offset(Range.end);
      if (!EndOffset)
        return EndOffset.takeError();
      // Nothing to do if the token/name hasn't changed.
      auto CurName =
          InitialCode.substr(*StartOffset, *EndOffset - *StartOffset);
      if (CurName == NewName)
        continue;
      OccurrencesOffsets.emplace_back(*StartOffset, *EndOffset, NewName);
    }
  }

  tooling::Replacements RenameEdit;
  for (const auto &R : OccurrencesOffsets) {
    auto ByteLength = R.End - R.Start;
    if (auto Err = RenameEdit.add(
            tooling::Replacement(AbsFilePath, R.Start, ByteLength, R.NewName)))
      return std::move(Err);
  }
  return Edit(InitialCode, std::move(RenameEdit));
}

// Details:
//  - lex the draft code to get all rename candidates, this yields a superset
//    of candidates.
//  - apply range patching heuristics to generate "authoritative" occurrences,
//    cases we consider:
//      (a) index returns a subset of candidates, we use the indexed results.
//        - fully equal, we are sure the index is up-to-date
//        - proper subset, index is correct in most cases? there may be false
//          positives (e.g. candidates got appended), but rename is still safe
//      (b) index returns non-candidate results, we attempt to map the indexed
//          ranges onto candidates in a plausible way (e.g. guess that lines
//          were inserted). If such a "near miss" is found, the rename is still
//          possible
std::optional<std::vector<SymbolRange>>
adjustRenameRanges(llvm::StringRef DraftCode, const RenameSymbolName &Name,
                   std::vector<Range> Indexed, const LangOptions &LangOpts) {
  trace::Span Tracer("AdjustRenameRanges");
  assert(!Indexed.empty());
  assert(llvm::is_sorted(Indexed));
  std::vector<SymbolRange> Lexed =
      collectRenameIdentifierRanges(Name, DraftCode, LangOpts);
  llvm::sort(Lexed);
  return getMappedRanges(Indexed, Lexed);
}

std::optional<std::vector<SymbolRange>>
getMappedRanges(ArrayRef<Range> Indexed, ArrayRef<SymbolRange> Lexed) {
  trace::Span Tracer("GetMappedRanges");
  assert(!Indexed.empty());
  assert(llvm::is_sorted(Indexed));
  assert(llvm::is_sorted(Lexed));

  if (Indexed.size() > Lexed.size()) {
    vlog("The number of lexed occurrences is less than indexed occurrences");
    SPAN_ATTACH(
        Tracer, "error",
        "The number of lexed occurrences is less than indexed occurrences");
    return std::nullopt;
  }
  // Fast check for the special subset case.
  if (llvm::includes(Indexed, Lexed))
    return Lexed.vec();

  std::vector<size_t> Best;
  size_t BestCost = std::numeric_limits<size_t>::max();
  bool HasMultiple = false;
  std::vector<size_t> ResultStorage;
  int Fuel = 10000;
  findNearMiss(ResultStorage, Indexed, Lexed, 0, Fuel,
               [&](const std::vector<size_t> &Matched) {
                 size_t MCost =
                     renameRangeAdjustmentCost(Indexed, Lexed, Matched);
                 if (MCost < BestCost) {
                   BestCost = MCost;
                   Best = std::move(Matched);
                   HasMultiple = false; // reset
                   return;
                 }
                 if (MCost == BestCost)
                   HasMultiple = true;
               });
  if (HasMultiple) {
    vlog("The best near miss is not unique.");
    SPAN_ATTACH(Tracer, "error", "The best near miss is not unique");
    return std::nullopt;
  }
  if (Best.empty()) {
    vlog("Didn't find a near miss.");
    SPAN_ATTACH(Tracer, "error", "Didn't find a near miss");
    return std::nullopt;
  }
  std::vector<SymbolRange> Mapped;
  for (auto I : Best)
    Mapped.push_back(Lexed[I]);
  SPAN_ATTACH(Tracer, "mapped_ranges", static_cast<int64_t>(Mapped.size()));
  return Mapped;
}

// The cost is the sum of the implied edit sizes between successive diffs, only
// simple edits are considered:
//   - insert/remove a line (change line offset)
//   - insert/remove a character on an existing line (change column offset)
//
// Example I, total result is 1 + 1 = 2.
//   diff[0]: line + 1 <- insert a line before edit 0.
//   diff[1]: line + 1
//   diff[2]: line + 1
//   diff[3]: line + 2 <- insert a line before edits 2 and 3.
//
// Example II, total result is 1 + 1 + 1 = 3.
//   diff[0]: line + 1  <- insert a line before edit 0.
//   diff[1]: column + 1 <- remove a line between edits 0 and 1, and insert a
//   character on edit 1.
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed,
                                 ArrayRef<SymbolRange> Lexed,
                                 ArrayRef<size_t> MappedIndex) {
  assert(Indexed.size() == MappedIndex.size());
  assert(llvm::is_sorted(Indexed));
  assert(llvm::is_sorted(Lexed));

  int LastLine = -1;
  int LastDLine = 0, LastDColumn = 0;
  int Cost = 0;
  for (size_t I = 0; I < Indexed.size(); ++I) {
    int DLine =
        Indexed[I].start.line - Lexed[MappedIndex[I]].range().start.line;
    int DColumn = Indexed[I].start.character -
                  Lexed[MappedIndex[I]].range().start.character;
    int Line = Indexed[I].start.line;
    if (Line != LastLine)
      LastDColumn = 0; // column offsets don't carry cross lines.
    Cost += abs(DLine - LastDLine) + abs(DColumn - LastDColumn);
    std::tie(LastLine, LastDLine, LastDColumn) = std::tie(Line, DLine, DColumn);
  }
  return Cost;
}

} // namespace clangd
} // namespace clang
