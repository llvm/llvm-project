//===--- RemoveUsingNamespace.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AST.h"
#include "FindTarget.h"
#include "Selection.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"

namespace clang {
namespace clangd {
namespace {
/// Removes the 'using namespace' under the cursor and qualifies all accesses in
/// the current file. E.g.,
///   using namespace std;
///   vector<int> foo(std::map<int, int>);
/// Would become:
///   std::vector<int> foo(std::map<int, int>);
/// Currently limited to using namespace directives inside global namespace to
/// simplify implementation. Also the namespace must not contain using
/// directives.
class RemoveUsingNamespace : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override {
    return "Remove using namespace, re-qualify names instead";
  }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  const UsingDirectiveDecl *TargetDirective = nullptr;
};
REGISTER_TWEAK(RemoveUsingNamespace)

class FindSameUsings : public RecursiveASTVisitor<FindSameUsings> {
public:
  FindSameUsings(const UsingDirectiveDecl &Target,
                 std::vector<const UsingDirectiveDecl *> &Results)
      : TargetNS(Target.getNominatedNamespace()),
        TargetCtx(Target.getDeclContext()), Results(Results) {}

  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    if (D->getNominatedNamespace() != TargetNS ||
        D->getDeclContext() != TargetCtx)
      return true;
    Results.push_back(D);
    return true;
  }

private:
  const NamespaceDecl *TargetNS;
  const DeclContext *TargetCtx;
  std::vector<const UsingDirectiveDecl *> &Results;
};

/// Produce edit removing 'using namespace xxx::yyy' and the trailing semicolon.
llvm::Expected<tooling::Replacement>
removeUsingDirective(ASTContext &Ctx, const UsingDirectiveDecl *D) {
  auto &SM = Ctx.getSourceManager();
  llvm::Optional<Token> NextTok =
      Lexer::findNextToken(D->getEndLoc(), SM, Ctx.getLangOpts());
  if (!NextTok || NextTok->isNot(tok::semi))
    return error("no semicolon after using-directive");
  // FIXME: removing the semicolon may be invalid in some obscure cases, e.g.
  //        if (x) using namespace std; else using namespace bar;
  return tooling::Replacement(
      SM,
      CharSourceRange::getTokenRange(D->getBeginLoc(), NextTok->getLocation()),
      "", Ctx.getLangOpts());
}

// Returns true iff the parent of the Node is a TUDecl.
bool isTopLevelDecl(const SelectionTree::Node *Node) {
  return Node->Parent && Node->Parent->ASTNode.get<TranslationUnitDecl>();
}

// Returns the first visible context that contains this DeclContext.
// For example: Returns ns1 for S1 and a.
// namespace ns1 {
// inline namespace ns2 { struct S1 {}; }
// enum E { a, b, c, d };
// }
const DeclContext *visibleContext(const DeclContext *D) {
  while (D->isInlineNamespace() || D->isTransparentContext())
    D = D->getParent();
  return D;
}

bool RemoveUsingNamespace::prepare(const Selection &Inputs) {
  // Find the 'using namespace' directive under the cursor.
  auto *CA = Inputs.ASTSelection.commonAncestor();
  if (!CA)
    return false;
  TargetDirective = CA->ASTNode.get<UsingDirectiveDecl>();
  if (!TargetDirective)
    return false;
  if (!isa<Decl>(TargetDirective->getDeclContext()))
    return false;
  // FIXME: Unavailable for namespaces containing using-namespace decl.
  // It is non-trivial to deal with cases where identifiers come from the inner
  // namespace. For example map has to be changed to aa::map.
  // namespace aa {
  //   namespace bb { struct map {}; }
  //   using namespace bb;
  // }
  // using namespace a^a;
  // int main() { map m; }
  // We need to make this aware of the transitive using-namespace decls.
  if (!TargetDirective->getNominatedNamespace()->using_directives().empty())
    return false;
  return isTopLevelDecl(CA);
}

Expected<Tweak::Effect> RemoveUsingNamespace::apply(const Selection &Inputs) {
  auto &Ctx = Inputs.AST->getASTContext();
  auto &SM = Ctx.getSourceManager();
  // First, collect *all* using namespace directives that redeclare the same
  // namespace.
  std::vector<const UsingDirectiveDecl *> AllDirectives;
  FindSameUsings(*TargetDirective, AllDirectives).TraverseAST(Ctx);

  SourceLocation FirstUsingDirectiveLoc;
  for (auto *D : AllDirectives) {
    if (FirstUsingDirectiveLoc.isInvalid() ||
        SM.isBeforeInTranslationUnit(D->getBeginLoc(), FirstUsingDirectiveLoc))
      FirstUsingDirectiveLoc = D->getBeginLoc();
  }

  // Collect all references to symbols from the namespace for which we're
  // removing the directive.
  std::vector<SourceLocation> IdentsToQualify;
  for (auto &D : Inputs.AST->getLocalTopLevelDecls()) {
    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (Ref.Qualifier)
            return; // This reference is already qualified.

          for (auto *T : Ref.Targets) {
            if (!visibleContext(T->getDeclContext())
                     ->Equals(TargetDirective->getNominatedNamespace()))
              return;
            // Avoid adding qualifiers before operators, e.g.
            //   using namespace std;
            //   cout << "foo"; // Must not changed to std::cout std:: << "foo"
            // FIXME: User-defined literals are not handled
            if (T->isInIdentifierNamespace(
                    Decl::IdentifierNamespace::IDNS_NonMemberOperator))
              return;
          }
          SourceLocation Loc = Ref.NameLoc;
          if (Loc.isMacroID()) {
            // Avoid adding qualifiers before macro expansions, it's probably
            // incorrect, e.g.
            //   namespace std { int foo(); }
            //   #define FOO 1 + foo()
            //   using namespace foo; // provides matrix
            //   auto x = FOO; // Must not changed to auto x = std::FOO
            if (!SM.isMacroArgExpansion(Loc))
              return; // FIXME: report a warning to the users.
            Loc = SM.getFileLoc(Ref.NameLoc);
          }
          assert(Loc.isFileID());
          if (SM.getFileID(Loc) != SM.getMainFileID())
            return; // FIXME: report these to the user as warnings?
          if (SM.isBeforeInTranslationUnit(Loc, FirstUsingDirectiveLoc))
            return; // Directive was not visible before this point.
          IdentsToQualify.push_back(Loc);
        },
        Inputs.AST->getHeuristicResolver());
  }
  // Remove duplicates.
  llvm::sort(IdentsToQualify);
  IdentsToQualify.erase(
      std::unique(IdentsToQualify.begin(), IdentsToQualify.end()),
      IdentsToQualify.end());

  // Produce replacements to remove the using directives.
  tooling::Replacements R;
  for (auto *D : AllDirectives) {
    auto RemoveUsing = removeUsingDirective(Ctx, D);
    if (!RemoveUsing)
      return RemoveUsing.takeError();
    if (auto Err = R.add(*RemoveUsing))
      return std::move(Err);
  }
  // Produce replacements to add the qualifiers.
  std::string Qualifier = printUsingNamespaceName(Ctx, *TargetDirective) + "::";
  for (auto Loc : IdentsToQualify) {
    if (auto Err = R.add(tooling::Replacement(Ctx.getSourceManager(), Loc,
                                              /*Length=*/0, Qualifier)))
      return std::move(Err);
  }
  return Effect::mainFileEdit(SM, std::move(R));
}

} // namespace
} // namespace clangd
} // namespace clang
