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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
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

namespace clang {
namespace clangd {
namespace {

llvm::Optional<std::string> filePath(const SymbolLocation &Loc,
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
  auto UD = std::find_if(Decls.begin(), Decls.end(), [](const NamedDecl *D) {
    return llvm::isa<UsingDecl>(D);
  });
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

llvm::Optional<ReasonToReject> renameable(const NamedDecl &RenameDecl,
                                          StringRef MainFilePath,
                                          const SymbolIndex *Index,
                                          const RenameOptions& Opts) {
  trace::Span Tracer("Renameable");
  if (!Opts.RenameVirtual) {
    if (const auto *S = llvm::dyn_cast<CXXMethodDecl>(&RenameDecl)) {
      if (S->isVirtual())
        return ReasonToReject::UnsupportedSymbol;
    }
  }
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
        if (ND != &RenamedDecl && ND->getName() == Name)
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
        if (Parameter->getName() == NewName)
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
      if (Parameter != &RenamedDecl && Parameter->getName() == NewName)
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
    if (D->getCanonicalDecl() != RenamedDecl.getCanonicalDecl())
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

static bool mayBeValidIdentifier(llvm::StringRef Ident) {
  assert(llvm::json::isUTF8(Ident));
  if (Ident.empty())
    return false;
  // We don't check all the rules for non-ascii characters (most are allowed).
  bool AllowDollar = true; // lenient
  if (llvm::isASCII(Ident.front()) &&
      !isAsciiIdentifierStart(Ident.front(), AllowDollar))
    return false;
  for (char C : Ident) {
    if (llvm::isASCII(C) && !isAsciiIdentifierContinue(C, AllowDollar))
      return false;
  }
  return true;
}

// Check if we can rename the given RenameDecl into NewName.
// Return details if the rename would produce a conflict.
llvm::Optional<InvalidName> checkName(const NamedDecl &RenameDecl,
                                      llvm::StringRef NewName) {
  trace::Span Tracer("CheckName");
  static constexpr trace::Metric InvalidNameMetric(
      "rename_name_invalid", trace::Metric::Counter, "invalid_kind");
  auto &ASTCtx = RenameDecl.getASTContext();
  llvm::Optional<InvalidName> Result;
  if (isKeyword(NewName, ASTCtx.getLangOpts()))
    Result = InvalidName{InvalidName::Keywords, NewName.str()};
  else if (!mayBeValidIdentifier(NewName))
    Result = InvalidName{InvalidName::BadIdentifier, NewName.str()};
  else {
    // Name conflict detection.
    // Function conflicts are subtle (overloading), so ignore them.
    if (RenameDecl.getKind() != Decl::Function) {
      if (auto *Conflict = lookupSiblingWithName(ASTCtx, RenameDecl, NewName))
        Result = InvalidName{
            InvalidName::Conflict,
            Conflict->getLocation().printToString(ASTCtx.getSourceManager())};
    }
  }
  if (Result)
    InvalidNameMetric.record(1, toString(Result->K));
  return Result;
}

// AST-based rename, it renames all occurrences in the main file.
llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, const NamedDecl &RenameDecl,
                 llvm::StringRef NewName) {
  trace::Span Tracer("RenameWithinFile");
  const SourceManager &SM = AST.getSourceManager();

  tooling::Replacements FilteredChanges;
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
    if (auto Err = FilteredChanges.add(tooling::Replacement(
            SM, CharSourceRange::getTokenRange(RenameLoc), NewName)))
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
    Ranges.erase(std::unique(Ranges.begin(), Ranges.end()), Ranges.end());

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

    auto AffectedFileCode = (*ExpBuffer)->getBuffer();
    auto RenameRanges =
        adjustRenameRanges(AffectedFileCode, RenameDecl.getNameAsString(),
                           std::move(FileAndOccurrences.second),
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
        buildRenameEdit(FilePath, AffectedFileCode, *RenameRanges, NewName);
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
    ArrayRef<Range> LexedRest, int LexedIndex, int &Fuel,
    llvm::function_ref<void(const std::vector<size_t> &)> MatchedCB) {
  if (--Fuel < 0)
    return;
  if (IndexedRest.size() > LexedRest.size())
    return;
  if (IndexedRest.empty()) {
    MatchedCB(PartialMatch);
    return;
  }
  if (impliesSimpleEdit(IndexedRest.front().start, LexedRest.front().start)) {
    PartialMatch.push_back(LexedIndex);
    findNearMiss(PartialMatch, IndexedRest.drop_front(), LexedRest.drop_front(),
                 LexedIndex + 1, Fuel, MatchedCB);
    PartialMatch.pop_back();
  }
  findNearMiss(PartialMatch, IndexedRest, LexedRest.drop_front(),
               LexedIndex + 1, Fuel, MatchedCB);
}

} // namespace

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
  const auto *ID = RenameDecl.getIdentifier();
  if (!ID)
    return makeError(ReasonToReject::UnsupportedSymbol);
  if (ID->getName() == RInputs.NewName)
    return makeError(ReasonToReject::SameName);
  auto Invalid = checkName(RenameDecl, RInputs.NewName);
  if (Invalid)
    return makeError(std::move(*Invalid));

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

  // Check the rename-triggering location is actually being renamed.
  // This is a robustness check to avoid surprising rename results -- if the
  // the triggering location is not actually the name of the node we identified
  // (e.g. for broken code), then rename is likely not what users expect, so we
  // reject this kind of rename.
  auto StartOffset = positionToOffset(MainFileCode, CurrentIdentifier.start);
  auto EndOffset = positionToOffset(MainFileCode, CurrentIdentifier.end);
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
  RenameResult Result;
  Result.Target = CurrentIdentifier;
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
                                     std::vector<Range> Occurrences,
                                     llvm::StringRef NewName) {
  trace::Span Tracer("BuildRenameEdit");
  SPAN_ATTACH(Tracer, "file_path", AbsFilePath);
  SPAN_ATTACH(Tracer, "rename_occurrences",
              static_cast<int64_t>(Occurrences.size()));

  assert(llvm::is_sorted(Occurrences));
  assert(std::unique(Occurrences.begin(), Occurrences.end()) ==
             Occurrences.end() &&
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

  std::vector<std::pair</*start*/ size_t, /*end*/ size_t>> OccurrencesOffsets;
  for (const auto &R : Occurrences) {
    auto StartOffset = Offset(R.start);
    if (!StartOffset)
      return StartOffset.takeError();
    auto EndOffset = Offset(R.end);
    if (!EndOffset)
      return EndOffset.takeError();
    OccurrencesOffsets.push_back({*StartOffset, *EndOffset});
  }

  tooling::Replacements RenameEdit;
  for (const auto &R : OccurrencesOffsets) {
    auto ByteLength = R.second - R.first;
    if (auto Err = RenameEdit.add(
            tooling::Replacement(AbsFilePath, R.first, ByteLength, NewName)))
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
llvm::Optional<std::vector<Range>>
adjustRenameRanges(llvm::StringRef DraftCode, llvm::StringRef Identifier,
                   std::vector<Range> Indexed, const LangOptions &LangOpts) {
  trace::Span Tracer("AdjustRenameRanges");
  assert(!Indexed.empty());
  assert(llvm::is_sorted(Indexed));
  std::vector<Range> Lexed =
      collectIdentifierRanges(Identifier, DraftCode, LangOpts);
  llvm::sort(Lexed);
  return getMappedRanges(Indexed, Lexed);
}

llvm::Optional<std::vector<Range>> getMappedRanges(ArrayRef<Range> Indexed,
                                                   ArrayRef<Range> Lexed) {
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
  if (std::includes(Indexed.begin(), Indexed.end(), Lexed.begin(), Lexed.end()))
    return Indexed.vec();

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
  std::vector<Range> Mapped;
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
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed, ArrayRef<Range> Lexed,
                                 ArrayRef<size_t> MappedIndex) {
  assert(Indexed.size() == MappedIndex.size());
  assert(llvm::is_sorted(Indexed));
  assert(llvm::is_sorted(Lexed));

  int LastLine = -1;
  int LastDLine = 0, LastDColumn = 0;
  int Cost = 0;
  for (size_t I = 0; I < Indexed.size(); ++I) {
    int DLine = Indexed[I].start.line - Lexed[MappedIndex[I]].start.line;
    int DColumn =
        Indexed[I].start.character - Lexed[MappedIndex[I]].start.character;
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
