//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "abi_tag_on_virtual.hpp"

// This clang-tidy check ensures that we don't place an abi_tag attribute on
// virtual functions. This can happen by mistakenly applying a macro like
// _LIBCPP_HIDE_FROM_ABI on a virtual function.
//
// The problem is that arm64e pointer authentication extensions use the mangled
// name of the function to sign the function pointer in the vtable, which means
// that the ABI tag effectively influences how the pointers are signed.
//
// This can lead to PAC failures when passing an object that holds one of these
// pointers in its vtable across an ABI boundary if the two sides have been compiled
// with different versions of libc++: one side will sign the pointer using one function
// mangling (with one ABI tag), and the other side will authenticate the pointer expecting
// it to have a different mangled name due to the ABI tag being different, which will crash.
//
// This test ensures that we don't re-introduce this issue in the code base.

namespace libcpp {
abi_tag_on_virtual::abi_tag_on_virtual(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void abi_tag_on_virtual::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(cxxMethodDecl(isVirtual(), hasAttr(clang::attr::AbiTag)).bind("abi_tag_on_virtual"), this);
}

void abi_tag_on_virtual::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs<clang::CXXMethodDecl>("abi_tag_on_virtual"); call != nullptr) {
    diag(call->getBeginLoc(),
         "_LIBCPP_HIDE_FROM_ABI should not be used on virtual functions to avoid problems with pointer authentication. "
         "Use _LIBCPP_HIDE_FROM_ABI_VIRTUAL instead.");
  }
}
} // namespace libcpp
