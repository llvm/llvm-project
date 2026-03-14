//===---- ObjCModuleTest.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyTest.h"
#include "gtest/gtest.h"
#include "objc/ForbiddenSubclassingCheck.h"

using namespace clang::tidy::objc;

namespace clang {
namespace tidy {
namespace test {

TEST(ObjCForbiddenSubclassing, AllowedSubclass) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<ForbiddenSubclassingCheck>(
      "@interface Foo\n"
      "@end\n"
      "@interface Bar : Foo\n"
      "@end\n",
      &Errors,
      "input.m");
  EXPECT_EQ(0ul, Errors.size());
}

TEST(ObjCForbiddenSubclassing, ForbiddenSubclass) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<ForbiddenSubclassingCheck>(
      "@interface UIImagePickerController\n"
      "@end\n"
      "@interface Foo : UIImagePickerController\n"
      "@end\n",
      &Errors,
      "input.m");
  EXPECT_EQ(1ul, Errors.size());
  EXPECT_EQ(
      "Objective-C interface 'Foo' subclasses 'UIImagePickerController', which is not intended to be subclassed",
      Errors[0].Message.Message);
}

} // namespace test
} // namespace tidy
} // namespace clang
