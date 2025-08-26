//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "hide_from_abi.hpp"

namespace {
AST_MATCHER(clang::ClassTemplateDecl, hasFullSpecializations) { return !Node.specializations().empty(); }
AST_MATCHER(clang::CXXRecordDecl, isTrivial) { return Node.isTrivial(); }
} // namespace

namespace libcpp {
hide_from_abi::hide_from_abi(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void hide_from_abi::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;

  auto has_hide_from_abi_attr = anyOf(hasAttr(clang::attr::Visibility), hasAttr(clang::attr::AbiTag));

  finder->addMatcher(
      functionDecl(
          unless(anyOf(
              // These functions can't be marked `[[gnu::always_inline]]` for various reasons,
              // so we can't mark them `_LIBCPP_HIDE_FROM_ABI`. These functions are ignored in
              // all namespaces. Checking the qualified name is a lot harder and these names
              // should result in very few (if any) false-negatives. This is also just a
              // temporary work-around until we can mark functions as HIDE_FROM_ABI without
              // having to add `[[gnu::always_inline]]` with GCC.
              hasAnyName("__introsort",
                         "__inplace_merge",
                         "__libcpp_snprintf_l",
                         "__libcpp_asprintf_l",
                         "__libcpp_sscanf_l",
                         "__tree_sub_invariant",
                         "__stable_sort_move",
                         "__stable_sort",
                         "__stable_partition",
                         "__lock_first",
                         "__stable_partition_impl"),
              has_hide_from_abi_attr,
              cxxMethodDecl(), // We have explicitly instantiated classes and some of their methods don't have these attributes
              isDeleted(),
              isConsteval())),
          isDefinition())
          .bind("hide_from_abi_on_free_function"),
      this);

  auto on_trivial = allOf(
      unless(isImplicit()), isDefaulted(), unless(ofClass(hasAncestor(classTemplateDecl()))), ofClass(isTrivial()));

  // TODO: find a better way to check for explicit instantiations. Currently, every template that has a full
  // specialization is ignored. For example, vector is ignored because we instantiate vector<double>
  // in discrete_distribution.
  finder->addMatcher(
      cxxMethodDecl(
          unless(anyOf(
              has_hide_from_abi_attr,
              isDeleted(),
              isImplicit(),
              hasAncestor(cxxRecordDecl(isLambda())),
              ofClass(anyOf(hasAncestor(classTemplateDecl(hasFullSpecializations())),
                            hasAnyName("basic_filebuf",
                                       "basic_ifstream",
                                       "basic_ofstream", // These are in the dylib in ABIv2
                                       // TODO: fix the matcher to catch `sentry` instantiation.
                                       "sentry"))),
              isConsteval(),
              hasParent(classTemplateSpecializationDecl()),
              on_trivial)),
          isDefinition())
          .bind("hide_from_abi_on_member_function"),
      this);

  finder->addMatcher(
      cxxMethodDecl(has_hide_from_abi_attr, on_trivial).bind("hide_from_abi_on_defaulted_smf_in_trivial_class"), this);
}

void hide_from_abi::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs<clang::FunctionDecl>("hide_from_abi_on_free_function");
      call != nullptr) {
    diag(call->getBeginLoc(), "_LIBCPP_HIDE_FROM_ABI is missing");
  }

  // The rest gets ignored in C++03 because it is subtly different in some cases.
  // e.g. we change the definition of default constructors in some cases
  // TODO: check whether we can remove thse differences
  if (!result.Context->getLangOpts().CPlusPlus11)
    return;

  if (const auto* call = result.Nodes.getNodeAs<clang::CXXMethodDecl>("hide_from_abi_on_member_function");
      call != nullptr) {
    diag(call->getLocation(), "_LIBCPP_HIDE_FROM_ABI or _LIBCPP_HIDE_FROM_ABI_VIRTUAL is missing");
  }

  if (const auto* call =
          result.Nodes.getNodeAs<clang::CXXMethodDecl>("hide_from_abi_on_defaulted_smf_in_trivial_class");
      call != nullptr) {
    diag(call->getLocation(),
         "_LIBCPP_HIDE_FROM_ABI should not be used for special member functions in trivial classes");
  }
}
} // namespace libcpp
