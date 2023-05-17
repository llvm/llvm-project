//===-- ExpectedTypeTest.cpp  -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpectedTypes.h"
#include "ParsedAST.h"
#include "TestTU.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

using ::testing::Field;
using ::testing::Matcher;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

class ExpectedTypeConversionTest : public ::testing::Test {
protected:
  void build(llvm::StringRef Code) {
    assert(!AST && "AST built twice");
    AST = TestTU::withCode(Code).build();
  }

  const NamedDecl *decl(llvm::StringRef Name) { return &findDecl(*AST, Name); }

  QualType typeOf(llvm::StringRef Name) {
    return cast<ValueDecl>(decl(Name))->getType().getCanonicalType();
  }

  /// An overload for convenience.
  std::optional<OpaqueType> fromCompletionResult(const NamedDecl *D) {
    return OpaqueType::fromCompletionResult(
        astCtx(), CodeCompletionResult(D, CCP_Declaration));
  }

  /// A set of DeclNames whose type match each other computed by
  /// OpaqueType::fromCompletionResult.
  using EquivClass = std::set<std::string>;

  Matcher<std::map<std::string, EquivClass>>
  classesAre(llvm::ArrayRef<EquivClass> Classes) {
    using MapEntry = std::map<std::string, EquivClass>::value_type;

    std::vector<Matcher<MapEntry>> Elements;
    Elements.reserve(Classes.size());
    for (auto &Cls : Classes)
      Elements.push_back(Field(&MapEntry::second, Cls));
    return UnorderedElementsAreArray(Elements);
  }

  // Groups \p Decls into equivalence classes based on the result of
  // 'OpaqueType::fromCompletionResult'.
  std::map<std::string, EquivClass>
  buildEquivClasses(llvm::ArrayRef<llvm::StringRef> DeclNames) {
    std::map<std::string, EquivClass> Classes;
    for (llvm::StringRef Name : DeclNames) {
      auto Type = OpaqueType::fromType(astCtx(), typeOf(Name));
      Classes[std::string(Type->raw())].insert(std::string(Name));
    }
    return Classes;
  }

  ASTContext &astCtx() { return AST->getASTContext(); }

private:
  // Set after calling build().
  std::optional<ParsedAST> AST;
};

TEST_F(ExpectedTypeConversionTest, BasicTypes) {
  build(R"cpp(
    // ints.
    bool b;
    int i;
    unsigned int ui;
    long long ll;

    // floats.
    float f;
    double d;

    // pointers
    int* iptr;
    bool* bptr;

    // user-defined types.
    struct X {};
    X user_type;
  )cpp");

  EXPECT_THAT(buildEquivClasses({"b", "i", "ui", "ll", "f", "d", "iptr", "bptr",
                                 "user_type"}),
              classesAre({{"b"},
                          {"i", "ui", "ll"},
                          {"f", "d"},
                          {"iptr"},
                          {"bptr"},
                          {"user_type"}}));
}

TEST_F(ExpectedTypeConversionTest, ReferencesDontMatter) {
  build(R"cpp(
    int noref;
    int & ref = noref;
    const int & const_ref = noref;
    int && rv_ref = 10;
  )cpp");

  EXPECT_THAT(buildEquivClasses({"noref", "ref", "const_ref", "rv_ref"}),
              SizeIs(1));
}

TEST_F(ExpectedTypeConversionTest, ArraysDecay) {
  build(R"cpp(
     int arr[2];
     int (&arr_ref)[2] = arr;
     int *ptr;
  )cpp");

  EXPECT_THAT(buildEquivClasses({"arr", "arr_ref", "ptr"}), SizeIs(1));
}

TEST_F(ExpectedTypeConversionTest, FunctionReturns) {
  build(R"cpp(
     int returns_int();
     int* returns_ptr();

     int int_;
     int* int_ptr;
  )cpp");

  OpaqueType IntTy = *OpaqueType::fromType(astCtx(), typeOf("int_"));
  EXPECT_EQ(fromCompletionResult(decl("returns_int")), IntTy);

  OpaqueType IntPtrTy = *OpaqueType::fromType(astCtx(), typeOf("int_ptr"));
  EXPECT_EQ(fromCompletionResult(decl("returns_ptr")), IntPtrTy);
}

TEST_F(ExpectedTypeConversionTest, Templates) {
  build(R"cpp(
template <class T>
int* returns_not_dependent();
template <class T>
T* returns_dependent();

template <class T>
int* var_not_dependent = nullptr;
template <class T>
T* var_dependent = nullptr;

int* int_ptr_;
  )cpp");

  auto IntPtrTy = *OpaqueType::fromType(astCtx(), typeOf("int_ptr_"));
  EXPECT_EQ(fromCompletionResult(decl("returns_not_dependent")), IntPtrTy);
  EXPECT_EQ(fromCompletionResult(decl("returns_dependent")), std::nullopt);

  EXPECT_EQ(fromCompletionResult(decl("var_not_dependent")), IntPtrTy);
  EXPECT_EQ(fromCompletionResult(decl("var_dependent")), std::nullopt);
}

} // namespace
} // namespace clangd
} // namespace clang
