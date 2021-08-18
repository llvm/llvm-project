//===- unittests/AST/AttrTests.cpp --- Attribute tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/Basic/AttrKinds.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

TEST(Attr, Doc) {
  EXPECT_THAT(Attr::getDocumentation(attr::Used).str(),
              testing::HasSubstr("The compiler must emit the definition even "
                                 "if it appears to be unused"));
}

} // namespace
