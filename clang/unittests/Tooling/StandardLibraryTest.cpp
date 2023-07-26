//===- unittest/Tooling/StandardLibrary.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Contains;
using ::testing::ElementsAre;

namespace clang {
namespace tooling {
namespace {

const NamedDecl &lookup(TestAST &AST, llvm::StringRef Name) {
  TranslationUnitDecl *TU = AST.context().getTranslationUnitDecl();
  auto Result = TU->lookup(DeclarationName(&AST.context().Idents.get(Name)));
  assert(!Result.empty() && "Lookup failed");
  assert(Result.isSingleResult() && "Lookup returned multiple results");
  return *Result.front();
}

TEST(StdlibTest, All) {
  auto VectorH = stdlib::Header::named("<vector>");
  EXPECT_TRUE(VectorH);
  EXPECT_EQ(VectorH->name(), "<vector>");
  EXPECT_EQ(llvm::to_string(*VectorH), "<vector>");
  EXPECT_FALSE(stdlib::Header::named("HeadersTests.cpp"));

  EXPECT_TRUE(stdlib::Header::named("<vector>", stdlib::Lang::CXX));
  EXPECT_FALSE(stdlib::Header::named("<vector>", stdlib::Lang::C));

  auto Vector = stdlib::Symbol::named("std::", "vector");
  EXPECT_TRUE(Vector);
  EXPECT_EQ(Vector->scope(), "std::");
  EXPECT_EQ(Vector->name(), "vector");
  EXPECT_EQ(Vector->qualifiedName(), "std::vector");
  EXPECT_EQ(llvm::to_string(*Vector), "std::vector");
  EXPECT_FALSE(stdlib::Symbol::named("std::", "dongle"));
  EXPECT_FALSE(stdlib::Symbol::named("clang::", "ASTContext"));

  EXPECT_TRUE(stdlib::Symbol::named("std::", "vector", stdlib::Lang::CXX));
  EXPECT_FALSE(stdlib::Symbol::named("std::", "vector", stdlib::Lang::C));

  EXPECT_EQ(Vector->header(), *VectorH);
  EXPECT_THAT(Vector->headers(), ElementsAre(*VectorH));

  EXPECT_TRUE(stdlib::Symbol::named("std::", "get"));
  EXPECT_FALSE(stdlib::Symbol::named("std::", "get")->header());

  EXPECT_THAT(stdlib::Symbol::named("std::", "basic_iostream")->headers(),
              ElementsAre(stdlib::Header::named("<istream>"),
                          stdlib::Header::named("<iostream>"),
                          stdlib::Header::named("<iosfwd>")));
  EXPECT_THAT(stdlib::Symbol::named("std::", "size_t")->headers(),
              ElementsAre(stdlib::Header::named("<cstddef>"),
                          stdlib::Header::named("<cstdlib>"),
                          stdlib::Header::named("<cstring>"),
                          stdlib::Header::named("<cwchar>"),
                          stdlib::Header::named("<cuchar>"),
                          stdlib::Header::named("<ctime>"),
                          stdlib::Header::named("<cstdio>")));
  EXPECT_EQ(stdlib::Symbol::named("std::", "size_t")->header(),
            stdlib::Header::named("<cstddef>"));

  EXPECT_THAT(stdlib::Header::all(), Contains(*VectorH));
  EXPECT_THAT(stdlib::Symbol::all(), Contains(*Vector));
  EXPECT_TRUE(stdlib::Header::named("<stdint.h>", stdlib::Lang::CXX));
  EXPECT_FALSE(stdlib::Header::named("<ios646.h>", stdlib::Lang::CXX));
}

TEST(StdlibTest, Experimental) {
  EXPECT_FALSE(
      stdlib::Header::named("<experimental/filesystem>", stdlib::Lang::C));
  EXPECT_TRUE(
      stdlib::Header::named("<experimental/filesystem>", stdlib::Lang::CXX));

  auto Symbol = stdlib::Symbol::named("std::experimental::filesystem::",
                                      "system_complete");
  EXPECT_TRUE(Symbol);
  EXPECT_EQ(Symbol->scope(), "std::experimental::filesystem::");
  EXPECT_EQ(Symbol->name(), "system_complete");
  EXPECT_EQ(Symbol->header(),
            stdlib::Header::named("<experimental/filesystem>"));
  EXPECT_EQ(Symbol->qualifiedName(),
            "std::experimental::filesystem::system_complete");
}

TEST(StdlibTest, CCompat) {
  EXPECT_THAT(
      stdlib::Symbol::named("", "int16_t", stdlib::Lang::CXX)->headers(),
      ElementsAre(stdlib::Header::named("<cstdint>"),
                  stdlib::Header::named("<stdint.h>")));
  EXPECT_THAT(
      stdlib::Symbol::named("std::", "int16_t", stdlib::Lang::CXX)->headers(),
      ElementsAre(stdlib::Header::named("<cstdint>")));

  EXPECT_TRUE(stdlib::Header::named("<stdint.h>", stdlib::Lang::C));
  EXPECT_THAT(
      stdlib::Symbol::named("", "int16_t", stdlib::Lang::C)->headers(),
      ElementsAre(stdlib::Header::named("<stdint.h>", stdlib::Lang::C)));
  EXPECT_FALSE(stdlib::Symbol::named("std::", "int16_t", stdlib::Lang::C));
}

TEST(StdlibTest, Recognizer) {
  TestAST AST(R"cpp(
    namespace std {
    inline namespace inl {

    template <typename>
    struct vector { class nested {}; };

    class secret {};

    } // inl

    inline namespace __1 {
      namespace chrono {
        inline namespace chrono_inl {
        class system_clock {};
        } // chrono_inl
      } // chrono
    } // __1

    } // std

    // C Standard Library structure defined in <stdlib.h>
    struct div_t {};

    class vector {};
    std::vector<int> vec;
    std::vector<int>::nested nest;
    std::secret sec;
    std::chrono::system_clock clock;

    div_t div;
  )cpp");

  auto &VectorNonstd = lookup(AST, "vector");
  auto *Vec = cast<VarDecl>(lookup(AST, "vec")).getType()->getAsCXXRecordDecl();
  auto *Nest =
      cast<VarDecl>(lookup(AST, "nest")).getType()->getAsCXXRecordDecl();
  auto *Clock =
      cast<VarDecl>(lookup(AST, "clock")).getType()->getAsCXXRecordDecl();
  auto *Sec = cast<VarDecl>(lookup(AST, "sec")).getType()->getAsCXXRecordDecl();
  auto *CDivT =
      cast<VarDecl>(lookup(AST, "div")).getType()->getAsCXXRecordDecl();

  stdlib::Recognizer Recognizer;

  EXPECT_EQ(Recognizer(&VectorNonstd), std::nullopt);
  EXPECT_EQ(Recognizer(Vec), stdlib::Symbol::named("std::", "vector"));
  EXPECT_EQ(Recognizer(Vec),
            stdlib::Symbol::named("std::", "vector", stdlib::Lang::CXX));
  EXPECT_EQ(Recognizer(Nest), stdlib::Symbol::named("std::", "vector"));
  EXPECT_EQ(Recognizer(Clock),
            stdlib::Symbol::named("std::chrono::", "system_clock"));
  auto DivT = stdlib::Symbol::named("", "div_t", stdlib::Lang::CXX);
  EXPECT_TRUE(DivT);
  EXPECT_EQ(Recognizer(CDivT), DivT);
  EXPECT_EQ(Recognizer(Sec), std::nullopt);
}

TEST(StdlibTest, RecognizerForC99) {
  TestInputs Input("typedef char uint8_t;");
  Input.Language = TestLanguage::Lang_C99;
  TestAST AST(Input);

  auto &Uint8T = lookup(AST, "uint8_t");
  stdlib::Recognizer Recognizer;
  EXPECT_EQ(Recognizer(&Uint8T),
            stdlib::Symbol::named("", "uint8_t", stdlib::Lang::C));
}

} // namespace
} // namespace tooling
} // namespace clang
