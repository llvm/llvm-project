//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NamedParameterCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

// Types whose parameters do not need a name (tag dispatch types, category
// tags, etc.).
static constexpr StringRef DefaultIgnoredTypes =
    "std::adopt_lock_t;"
    "std::allocator_arg_t;"
    "std::bidirectional_iterator_tag;"
    "std::contiguous_iterator_tag;"
    "std::default_sentinel_t;"
    "std::defer_lock_t;"
    "std::destroying_delete_t;"
    "std::forward_iterator_tag;"
    "std::from_range_t;"
    "std::in_place_index_t;"
    "std::in_place_t;"
    "std::in_place_type_t;"
    "std::input_iterator_tag;"
    "std::nothrow_t;"
    "std::nostopstate_t;"
    "std::nullopt_t;"
    "std::output_iterator_tag;"
    "std::piecewise_construct_t;"
    "std::random_access_iterator_tag;"
    "std::sorted_equivalent_t;"
    "std::sorted_unique_t;"
    "std::try_to_lock_t;"
    "std::unexpect_t;"
    "std::unreachable_sentinel_t";

NamedParameterCheck::NamedParameterCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      InsertPlainNamesInForwardDecls(
          Options.get("InsertPlainNamesInForwardDecls", false)),
      IgnoredTypes(utils::options::parseStringList(
          Options.get("IgnoredTypes", DefaultIgnoredTypes))) {}

void NamedParameterCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "InsertPlainNamesInForwardDecls",
                InsertPlainNamesInForwardDecls);
  Options.store(Opts, "IgnoredTypes",
                utils::options::serializeStringList(IgnoredTypes));
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
    if (const auto *Typedef = Parm->getType()->getAs<TypedefType>())
      if (Typedef->getDecl()->getQualifiedNameAsString() == "testing::Unused")
        continue;

    // Skip std::nullptr_t.
    if (Parm->getType().getCanonicalType()->isNullPtrType())
      continue;

    // Skip the types configured by the IgnoredTypes option (e.g. standard
    // tag dispatch types).
    if (const auto *Record =
            Parm->getType().getCanonicalType()->getAsCXXRecordDecl()) {
      const std::string QName = Record->getQualifiedNameAsString();
      if (llvm::is_contained(IgnoredTypes, QName))
        continue;
    }

    // Look for comments. We explicitly want to allow idioms like
    // void foo(int /*unused*/)
    const char *Begin = SM.getCharacterData(Parm->getBeginLoc());
    const char *End = SM.getCharacterData(Parm->getLocation());
    const StringRef Data(Begin, End - Begin);
    if (Data.contains("/*"))
      continue;

    UnnamedParams.emplace_back(Function, I);
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
        const StringRef Name = OtherParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // If the definition has a named parameter use that name.
      if (Definition) {
        const ParmVarDecl *DefParm = Definition->getParamDecl(P.second);
        const StringRef Name = DefParm->getName();
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
