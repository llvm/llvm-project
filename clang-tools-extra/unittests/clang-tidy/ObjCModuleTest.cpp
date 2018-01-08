//===---- ObjCModuleTest.cpp - clang-tidy ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
