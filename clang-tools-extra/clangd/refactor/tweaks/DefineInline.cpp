//===--- DefineInline.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "FindTarget.h"
#include "Logger.h"
#include "Selection.h"
#include "SourceCode.h"
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Returns semicolon location for the given FD. Since AST doesn't contain that
// information, searches for a semicolon by lexing from end of function decl
// while skipping comments.
llvm::Optional<SourceLocation> getSemicolonForDecl(const FunctionDecl *FD) {
  const SourceManager &SM = FD->getASTContext().getSourceManager();
  const LangOptions &LangOpts = FD->getASTContext().getLangOpts();

  SourceLocation CurLoc = FD->getEndLoc();
  auto NextTok = Lexer::findNextToken(CurLoc, SM, LangOpts);
  if (!NextTok || !NextTok->is(tok::semi))
    return llvm::None;
  return NextTok->getLocation();
}

// Deduces the FunctionDecl from a selection. Requires either the function body
// or the function decl to be selected. Returns null if none of the above
// criteria is met.
const FunctionDecl *getSelectedFunction(const SelectionTree::Node *SelNode) {
  const ast_type_traits::DynTypedNode &AstNode = SelNode->ASTNode;
  if (const FunctionDecl *FD = AstNode.get<FunctionDecl>())
    return FD;
  if (AstNode.get<CompoundStmt>() &&
      SelNode->Selected == SelectionTree::Complete) {
    if (const SelectionTree::Node *P = SelNode->Parent)
      return P->ASTNode.get<FunctionDecl>();
  }
  return nullptr;
}

// Checks the decls mentioned in Source are visible in the context of Target.
// Achives that by checking declaraions occur before target location in
// translation unit or declared in the same class.
bool checkDeclsAreVisible(const llvm::DenseSet<const Decl *> &DeclRefs,
                          const FunctionDecl *Target, const SourceManager &SM) {
  SourceLocation TargetLoc = Target->getLocation();
  // To be used in visibility check below, decls in a class are visible
  // independent of order.
  const RecordDecl *Class = nullptr;
  if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(Target))
    Class = MD->getParent();

  for (const auto *DR : DeclRefs) {
    // Use canonical decl, since having one decl before target is enough.
    const Decl *D = DR->getCanonicalDecl();
    if (D == Target)
      continue;
    SourceLocation DeclLoc = D->getLocation();

    // FIXME: Allow declarations from different files with include insertion.
    if (!SM.isWrittenInSameFile(DeclLoc, TargetLoc))
      return false;

    // If declaration is before target, then it is visible.
    if (SM.isBeforeInTranslationUnit(DeclLoc, TargetLoc))
      continue;

    // Otherwise they need to be in same class
    if (!Class)
      return false;
    const RecordDecl *Parent = nullptr;
    if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(D))
      Parent = MD->getParent();
    else if (const auto *FD = llvm::dyn_cast<FieldDecl>(D))
      Parent = FD->getParent();
    if (Parent != Class)
      return false;
  }
  return true;
}

// Rewrites body of FD by re-spelling all of the names to make sure they are
// still valid in context of Target.
llvm::Expected<std::string> qualifyAllDecls(const FunctionDecl *FD,
                                            const FunctionDecl *Target) {
  // There are three types of spellings that needs to be qualified in a function
  // body:
  // - Types:       Foo                 -> ns::Foo
  // - DeclRefExpr: ns2::foo()          -> ns1::ns2::foo();
  // - UsingDecls:
  //    using ns2::foo      -> using ns1::ns2::foo
  //    using namespace ns2 -> using namespace ns1::ns2
  //    using ns3 = ns2     -> using ns3 = ns1::ns2
  //
  // Go over all references inside a function body to generate replacements that
  // will qualify those. So that body can be moved into an arbitrary file.
  // We perform the qualification by qualyfying the first type/decl in a
  // (un)qualified name. e.g:
  //    namespace a { namespace b { class Bar{}; void foo(); } }
  //    b::Bar x; -> a::b::Bar x;
  //    foo(); -> a::b::foo();

  auto *TargetContext = Target->getLexicalDeclContext();
  const SourceManager &SM = FD->getASTContext().getSourceManager();

  tooling::Replacements Replacements;
  bool HadErrors = false;
  findExplicitReferences(FD->getBody(), [&](ReferenceLoc Ref) {
    // Since we want to qualify only the first qualifier, skip names with a
    // qualifier.
    if (Ref.Qualifier)
      return;
    // There might be no decl in dependent contexts, there's nothing much we can
    // do in such cases.
    if (Ref.Targets.empty())
      return;
    // Do not qualify names introduced by macro expansions.
    if (Ref.NameLoc.isMacroID())
      return;

    for (const NamedDecl *ND : Ref.Targets) {
      if (ND->getDeclContext() != Ref.Targets.front()->getDeclContext()) {
        elog("define inline: Targets from multiple contexts: {0}, {1}",
             printQualifiedName(*Ref.Targets.front()), printQualifiedName(*ND));
        HadErrors = true;
        return;
      }
    }
    // All Targets are in the same scope, so we can safely chose first one.
    const NamedDecl *ND = Ref.Targets.front();
    // Skip anything from a non-namespace scope, these can be:
    // - Function or Method scopes, which means decl is local and doesn't need
    //   qualification.
    // - From Class/Struct/Union scope, which again doesn't need any qualifiers,
    //   rather the left side of it requires qualification, like:
    //   namespace a { class Bar { public: static int x; } }
    //   void foo() { Bar::x; }
    //                ~~~~~ -> we need to qualify Bar not x.
    if (!ND->getDeclContext()->isNamespace())
      return;

    const std::string Qualifier = getQualification(
        FD->getASTContext(), TargetContext, Target->getBeginLoc(), ND);
    if (auto Err = Replacements.add(
            tooling::Replacement(SM, Ref.NameLoc, 0, Qualifier))) {
      HadErrors = true;
      elog("define inline: Failed to add quals: {0}", std::move(Err));
    }
  });

  if (HadErrors) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "define inline: Failed to compute qualifiers see logs for details.");
  }

  // Get new begin and end positions for the qualified body.
  auto OrigBodyRange = toHalfOpenFileRange(
      SM, FD->getASTContext().getLangOpts(), FD->getBody()->getSourceRange());
  if (!OrigBodyRange)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Couldn't get range func body.");

  unsigned BodyBegin = SM.getFileOffset(OrigBodyRange->getBegin());
  unsigned BodyEnd = Replacements.getShiftedCodePosition(
      SM.getFileOffset(OrigBodyRange->getEnd()));

  // Trim the result to function body.
  auto QualifiedFunc = tooling::applyAllReplacements(
      SM.getBufferData(SM.getFileID(OrigBodyRange->getBegin())), Replacements);
  if (!QualifiedFunc)
    return QualifiedFunc.takeError();
  return QualifiedFunc->substr(BodyBegin, BodyEnd - BodyBegin + 1);
}

/// Generates Replacements for changing template and function parameter names in
/// \p Dest to be the same as in \p Source.
llvm::Expected<tooling::Replacements>
renameParameters(const FunctionDecl *Dest, const FunctionDecl *Source) {
  llvm::DenseMap<const Decl *, std::string> ParamToNewName;
  llvm::DenseMap<const NamedDecl *, std::vector<SourceLocation>> RefLocs;
  auto HandleParam = [&](const NamedDecl *DestParam,
                         const NamedDecl *SourceParam) {
    // No need to rename if parameters already have the same name.
    if (DestParam->getName() == SourceParam->getName())
      return;
    std::string NewName;
    // Unnamed parameters won't be visited in findExplicitReferences. So add
    // them here.
    if (DestParam->getName().empty()) {
      RefLocs[DestParam].push_back(DestParam->getLocation());
      // If decl is unnamed in destination we pad the new name to avoid gluing
      // with previous token, e.g. foo(int^) shouldn't turn into foo(intx).
      NewName = " ";
    }
    NewName.append(SourceParam->getName());
    ParamToNewName[DestParam->getCanonicalDecl()] = std::move(NewName);
  };

  // Populate mapping for template parameters.
  auto *DestTempl = Dest->getDescribedFunctionTemplate();
  auto *SourceTempl = Source->getDescribedFunctionTemplate();
  assert(bool(DestTempl) == bool(SourceTempl));
  if (DestTempl) {
    const auto *DestTPL = DestTempl->getTemplateParameters();
    const auto *SourceTPL = SourceTempl->getTemplateParameters();
    assert(DestTPL->size() == SourceTPL->size());

    for (size_t I = 0, EP = DestTPL->size(); I != EP; ++I)
      HandleParam(DestTPL->getParam(I), SourceTPL->getParam(I));
  }

  // Populate mapping for function params.
  assert(Dest->param_size() == Source->param_size());
  for (size_t I = 0, E = Dest->param_size(); I != E; ++I)
    HandleParam(Dest->getParamDecl(I), Source->getParamDecl(I));

  const SourceManager &SM = Dest->getASTContext().getSourceManager();
  const LangOptions &LangOpts = Dest->getASTContext().getLangOpts();
  // Collect other references in function signature, i.e parameter types and
  // default arguments.
  findExplicitReferences(
      // Use function template in case of templated functions to visit template
      // parameters.
      DestTempl ? llvm::dyn_cast<Decl>(DestTempl) : llvm::dyn_cast<Decl>(Dest),
      [&](ReferenceLoc Ref) {
        if (Ref.Targets.size() != 1)
          return;
        const auto *Target =
            llvm::cast<NamedDecl>(Ref.Targets.front()->getCanonicalDecl());
        auto It = ParamToNewName.find(Target);
        if (It == ParamToNewName.end())
          return;
        RefLocs[Target].push_back(Ref.NameLoc);
      });

  // Now try to generate edits for all the refs.
  tooling::Replacements Replacements;
  for (auto &Entry : RefLocs) {
    const auto *OldDecl = Entry.first;
    llvm::StringRef OldName = OldDecl->getName();
    llvm::StringRef NewName = ParamToNewName[OldDecl];
    for (SourceLocation RefLoc : Entry.second) {
      CharSourceRange ReplaceRange;
      // In case of unnamed parameters, we have an empty char range, whereas we
      // have a tokenrange at RefLoc with named parameters.
      if (OldName.empty())
        ReplaceRange = CharSourceRange::getCharRange(RefLoc, RefLoc);
      else
        ReplaceRange = CharSourceRange::getTokenRange(RefLoc, RefLoc);
      // If occurence is coming from a macro expansion, try to get back to the
      // file range.
      if (RefLoc.isMacroID()) {
        ReplaceRange = Lexer::makeFileCharRange(ReplaceRange, SM, LangOpts);
        // Bail out if we need to replace macro bodies.
        if (ReplaceRange.isInvalid()) {
          auto Err = llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "Cant rename parameter inside macro body.");
          elog("define inline: {0}", Err);
          return std::move(Err);
        }
      }

      if (auto Err = Replacements.add(
              tooling::Replacement(SM, ReplaceRange, NewName))) {
        elog("define inline: Couldn't replace parameter name for {0} to {1}: "
             "{2}",
             OldName, NewName, Err);
        return std::move(Err);
      }
    }
  }
  return Replacements;
}

// Returns the canonical declaration for the given FunctionDecl. This will
// usually be the first declaration in current translation unit with the
// exception of template specialization.
// For those we return first declaration different than the canonical one.
// Because canonical declaration points to template decl instead of
// specialization.
const FunctionDecl *findTarget(const FunctionDecl *FD) {
  auto CanonDecl = FD->getCanonicalDecl();
  if (!FD->isFunctionTemplateSpecialization())
    return CanonDecl;
  // For specializations CanonicalDecl is the TemplatedDecl, which is not the
  // target we want to inline into. Instead we traverse previous decls to find
  // the first forward decl for this specialization.
  auto PrevDecl = FD;
  while (PrevDecl->getPreviousDecl() != CanonDecl) {
    PrevDecl = PrevDecl->getPreviousDecl();
    assert(PrevDecl && "Found specialization without template decl");
  }
  return PrevDecl;
}

// Returns the begining location for a FunctionDecl. Returns location of
// template keyword for templated functions.
const SourceLocation getBeginLoc(const FunctionDecl *FD) {
  // Include template parameter list.
  if (auto *FTD = FD->getDescribedFunctionTemplate())
    return FTD->getBeginLoc();
  return FD->getBeginLoc();
}

/// Moves definition of a function/method to its declaration location.
/// Before:
/// a.h:
///   void foo();
///
/// a.cc:
///   void foo() { return; }
///
/// ------------------------
/// After:
/// a.h:
///   void foo() { return; }
///
/// a.cc:
///
class DefineInline : public Tweak {
public:
  const char *id() const override final;

  Intent intent() const override { return Intent::Refactor; }
  std::string title() const override {
    return "Move function body to declaration";
  }

  // Returns true when selection is on a function definition that does not
  // make use of any internal symbols.
  bool prepare(const Selection &Sel) override {
    const SelectionTree::Node *SelNode = Sel.ASTSelection.commonAncestor();
    if (!SelNode)
      return false;
    Source = getSelectedFunction(SelNode);
    if (!Source || !Source->hasBody())
      return false;
    // Only the last level of template parameter locations are not kept in AST,
    // so if we are inlining a method that is in a templated class, there is no
    // way to verify template parameter names. Therefore we bail out.
    if (auto *MD = llvm::dyn_cast<CXXMethodDecl>(Source)) {
      if (MD->getParent()->isTemplated())
        return false;
    }
    // If function body starts or ends inside a macro, we refuse to move it into
    // declaration location.
    if (Source->getBody()->getBeginLoc().isMacroID() ||
        Source->getBody()->getEndLoc().isMacroID())
      return false;

    Target = findTarget(Source);
    if (Target == Source) {
      // The only declaration is Source. No other declaration to move function
      // body.
      // FIXME: If we are in an implementation file, figure out a suitable
      // location to put declaration. Possibly using other declarations in the
      // AST.
      return false;
    }

    // Check if the decls referenced in function body are visible in the
    // declaration location.
    if (!checkDeclsAreVisible(getNonLocalDeclRefs(Sel.AST, Source), Target,
                              Sel.AST.getSourceManager()))
      return false;

    return true;
  }

  Expected<Effect> apply(const Selection &Sel) override {
    const auto &AST = Sel.AST.getASTContext();
    const auto &SM = AST.getSourceManager();

    auto Semicolon = getSemicolonForDecl(Target);
    if (!Semicolon) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Couldn't find semicolon for target declaration.");
    }

    auto ParamReplacements = renameParameters(Target, Source);
    if (!ParamReplacements)
      return ParamReplacements.takeError();

    auto QualifiedBody = qualifyAllDecls(Source, Target);
    if (!QualifiedBody)
      return QualifiedBody.takeError();

    const tooling::Replacement SemicolonToFuncBody(SM, *Semicolon, 1,
                                                   *QualifiedBody);
    auto DefRange = toHalfOpenFileRange(
        SM, AST.getLangOpts(),
        SM.getExpansionRange(CharSourceRange::getCharRange(getBeginLoc(Source),
                                                           Source->getEndLoc()))
            .getAsRange());
    if (!DefRange) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't get range for the source.");
    }
    unsigned int SourceLen = SM.getFileOffset(DefRange->getEnd()) -
                             SM.getFileOffset(DefRange->getBegin());
    const tooling::Replacement DeleteFuncBody(SM, DefRange->getBegin(),
                                              SourceLen, "");

    llvm::SmallVector<std::pair<std::string, Edit>, 2> Edits;
    // Edit for Target.
    auto FE = Effect::fileEdit(
        SM, SM.getFileID(*Semicolon),
        tooling::Replacements(SemicolonToFuncBody).merge(*ParamReplacements));
    if (!FE)
      return FE.takeError();
    Edits.push_back(std::move(*FE));

    // Edit for Source.
    if (!SM.isWrittenInSameFile(DefRange->getBegin(),
                                SM.getExpansionLoc(Target->getBeginLoc()))) {
      // Generate a new edit if the Source and Target are in different files.
      auto FE = Effect::fileEdit(SM, SM.getFileID(Sel.Cursor),
                                 tooling::Replacements(DeleteFuncBody));
      if (!FE)
        return FE.takeError();
      Edits.push_back(std::move(*FE));
    } else {
      // Merge with previous edit if they are in the same file.
      if (auto Err = Edits.front().second.Replacements.add(DeleteFuncBody))
        return std::move(Err);
    }

    Effect E;
    for (auto &Pair : Edits)
      E.ApplyEdits.try_emplace(std::move(Pair.first), std::move(Pair.second));
    return E;
  }

private:
  const FunctionDecl *Source = nullptr;
  const FunctionDecl *Target = nullptr;
};

REGISTER_TWEAK(DefineInline)

} // namespace
} // namespace clangd
} // namespace clang
