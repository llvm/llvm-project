//=== unittests/CodeGen/DemangleTrapReasonInDebugInfo.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace clang::CodeGen;

void CheckValidCommon(llvm::StringRef FuncName, const char *ExpectedCategory,
                      const char *ExpectedMessage) {
  auto MaybeTrapReason = DemangleTrapReasonInDebugInfo(FuncName);
  ASSERT_TRUE(MaybeTrapReason.has_value());
  auto [Category, Message] = MaybeTrapReason.value();
  ASSERT_STREQ(Category.str().c_str(), ExpectedCategory);
  ASSERT_STREQ(Message.str().c_str(), ExpectedMessage);
}

void CheckInvalidCommon(llvm::StringRef FuncName) {
  auto MaybeTrapReason = DemangleTrapReasonInDebugInfo(FuncName);
  ASSERT_TRUE(!MaybeTrapReason.has_value());
}

TEST(DemangleTrapReasonInDebugInfo, Valid) {
  std::string FuncName(ClangTrapPrefix);
  FuncName += "$trap category$trap message";
  CheckValidCommon(FuncName, "trap category", "trap message");
}

TEST(DemangleTrapReasonInDebugInfo, ValidEmptyCategory) {
  std::string FuncName(ClangTrapPrefix);
  FuncName += "$$trap message";
  CheckValidCommon(FuncName, "", "trap message");
}

TEST(DemangleTrapReasonInDebugInfo, ValidEmptyMessage) {
  std::string FuncName(ClangTrapPrefix);
  FuncName += "$trap category$";
  CheckValidCommon(FuncName, "trap category", "");
}

TEST(DemangleTrapReasonInDebugInfo, ValidAllEmpty) {
  //  `__builtin_verbose_trap` actually allows this
  // currently. However, we should probably disallow this in Sema because having
  // an empty category and message completely defeats the point of using the
  // builtin (#165981).
  std::string FuncName(ClangTrapPrefix);
  FuncName += "$$";
  CheckValidCommon(FuncName, "", "");
}

TEST(DemangleTrapReasonInDebugInfo, InvalidOnlyPrefix) {
  std::string FuncName(ClangTrapPrefix);
  CheckInvalidCommon(FuncName);
}

TEST(DemangleTrapReasonInDebugInfo, Invalid) {
  std::string FuncName("foo");
  CheckInvalidCommon(FuncName);
}

TEST(DemangleTrapReasonInDebugInfo, InvalidEmpty) { CheckInvalidCommon(""); }
