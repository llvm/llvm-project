//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NamedParameterCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

NamedParameterCheck::NamedParameterCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      InsertPlainNamesInForwardDecls(
          Options.get("InsertPlainNamesInForwardDecls", false)) {}

void NamedParameterCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "InsertPlainNamesInForwardDecls",
                InsertPlainNamesInForwardDecls);
}

void NamedParameterCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(functionDecl().bind("decl"), this);
}

void NamedParameterCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("decl");
  SmallVector<std::pair<const FunctionDecl *, unsigned>, 4> UnnamedParams;

  // Ignore declarations without a definition if we're not dealing with an
  // overriden method.
  const FunctionDecl *Definition = nullptr;
  if ((!Function->isDefined(Definition) || Function->isDefaulted() ||
       Definition->isDefaulted() || Function->isDeleted()) &&
      (!isa<CXXMethodDecl>(Function) ||
       cast<CXXMethodDecl>(Function)->size_overridden_methods() == 0))
    return;

  // TODO: Handle overloads.
  // TODO: We could check that all redeclarations use the same name for
  //       arguments in the same position.
  for (unsigned I = 0, E = Function->getNumParams(); I != E; ++I) {
    const ParmVarDecl *Parm = Function->getParamDecl(I);
    if (Parm->isImplicit())
      continue;
    // Look for unnamed parameters.
    if (!Parm->getName().empty())
      continue;

    // Don't warn on the dummy argument on post-inc and post-dec operators.
    if ((Function->getOverloadedOperator() == OO_PlusPlus ||
         Function->getOverloadedOperator() == OO_MinusMinus) &&
        Parm->getType()->isSpecificBuiltinType(BuiltinType::Int))
      continue;

    // Sanity check the source locations.
    if (!Parm->getLocation().isValid() || Parm->getLocation().isMacroID() ||
        !SM.isWrittenInSameFile(Parm->getBeginLoc(), Parm->getLocation()))
      continue;

    // Skip gmock testing::Unused parameters.
    if (const auto *Typedef = Parm->getType()->getAs<clang::TypedefType>())
      if (Typedef->getDecl()->getQualifiedNameAsString() == "testing::Unused")
        continue;

    // Skip std::nullptr_t.
    if (Parm->getType().getCanonicalType()->isNullPtrType())
      continue;

    // Look for comments. We explicitly want to allow idioms like
    // void foo(int /*unused*/)
    const char *Begin = SM.getCharacterData(Parm->getBeginLoc());
    const char *End = SM.getCharacterData(Parm->getLocation());
    StringRef Data(Begin, End - Begin);
    if (Data.contains("/*"))
      continue;

    UnnamedParams.push_back(std::make_pair(Function, I));
  }

  // Emit only one warning per function but fixits for all unnamed parameters.
  if (!UnnamedParams.empty()) {
    const ParmVarDecl *FirstParm =
        UnnamedParams.front().first->getParamDecl(UnnamedParams.front().second);
    auto D = diag(FirstParm->getLocation(),
                  "all parameters should be named in a function");

    for (auto P : UnnamedParams) {
      // Fallback to an unused marker.
      static constexpr StringRef FallbackName = "unused";
      StringRef NewName = FallbackName;

      // If the method is overridden, try to copy the name from the base method
      // into the overrider.
      const auto *M = dyn_cast<CXXMethodDecl>(P.first);
      if (M && M->size_overridden_methods() > 0) {
        const ParmVarDecl *OtherParm =
            (*M->begin_overridden_methods())->getParamDecl(P.second);
        StringRef Name = OtherParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // If the definition has a named parameter use that name.
      if (Definition) {
        const ParmVarDecl *DefParm = Definition->getParamDecl(P.second);
        StringRef Name = DefParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // Now insert the fix. Note that getLocation() points to the place
      // where the name would be, this allows us to also get complex cases like
      // function pointers right.
      const ParmVarDecl *Parm = P.first->getParamDecl(P.second);

      // The fix depends on the InsertPlainNamesInForwardDecls option,
      // whether this is a forward declaration and whether the parameter has
      // a real name.
      const bool IsForwardDeclaration = (!Definition || Function != Definition);
      if (InsertPlainNamesInForwardDecls && IsForwardDeclaration &&
          NewName != FallbackName) {
        // For forward declarations with InsertPlainNamesInForwardDecls enabled,
        // insert the parameter name without comments.
        D << FixItHint::CreateInsertion(Parm->getLocation(),
                                        " " + NewName.str());
      } else {
        D << FixItHint::CreateInsertion(Parm->getLocation(),
                                        " /*" + NewName.str() + "*/");
      }
    }
  }
}

} // namespace clang::tidy::readability
