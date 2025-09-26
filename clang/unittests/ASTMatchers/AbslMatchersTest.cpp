//===- unittests/ASTMatchers/GTestMatchersTest.cpp - GTest matcher unit tests //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/AbslMatchers.h"
#include "ASTMatchersTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace ast_matchers {
namespace absl_matchers {

constexpr llvm::StringLiteral abslMockDecls = R"cc(
namespace absl {
template <typename T> class StatusOr {};
class Status {};
}

)cc";

static auto wrapAbsl(llvm::StringRef Input) { return abslMockDecls + Input; }

TEST(AbslMatchersTest, StatusOrDecl) {
  DeclarationMatcher StatusOrDecl =
      varDecl(hasType(qualType(hasDeclaration(statusOrClass()))));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      wrapAbsl("void Test() { absl::StatusOr<int> X; }"), StatusOrDecl,
      std::make_unique<VerifyIdIsBoundTo<Type>>("StatusOrValueType")));
  EXPECT_TRUE(notMatches(wrapAbsl("void Test() { int X; }"), StatusOrDecl));
  EXPECT_TRUE(notMatches(wrapAbsl(R"cc(
    namespace foo { namespace absl {
    template <typename T> class StatusOr {};
    }}
    void Test() { foo::absl::StatusOr<int> X; }
    )cc"),
                         StatusOrDecl));
}

TEST(AbslMatchersTest, StatusDecl) {
  DeclarationMatcher StatusDecl =
      varDecl(hasType(recordType(hasDeclaration(statusClass()))));
  EXPECT_TRUE(matches(wrapAbsl("void Test() { absl::Status X; }"), StatusDecl));
  EXPECT_TRUE(notMatches(wrapAbsl("void Test() { int X; }"), StatusDecl));
  EXPECT_TRUE(notMatches(wrapAbsl(R"cc(
    namespace foo { namespace absl {
    class Status {};
    }}
    void Test() { foo::absl::Status X; }
    )cc"),
                         StatusDecl));
}

} // end namespace absl_matchers
} // end namespace ast_matchers
} // end namespace clang
