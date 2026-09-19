//=== unittests/CodeGen/CGDebugInfoTest.cpp - CGDebugInfo tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestCompiler.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class StandardLayoutUnionDebugInfoTest : public ::testing::Test {
protected:
  TestCompiler Compiler;
  DebugInfoFinder Finder;

  static clang::CodeGenOptions getCodeGenOpts() {
    clang::CodeGenOptions CGOpts;
    CGOpts.setDebugInfo(llvm::codegenoptions::DebugInfoConstructor);
    return CGOpts;
  }

  static clang::LangOptions getLangOpts() {
    clang::LangOptions LO;
    LO.CPlusPlus = LO.CPlusPlus11 = 1;
    return LO;
  }

  StandardLayoutUnionDebugInfoTest()
      : Compiler(getLangOpts(), getCodeGenOpts()) {}

  void compile(const char *Code) {
    Compiler.init(Code);
    Finder.reset();
    Finder.processModule(*Compiler.compileModule());
  }

  const DICompositeType *findCompositeType(StringRef Name) const {
    for (DIType *T : Finder.types()) {
      if (T->getName() != Name)
        continue;
      while (auto *DT = dyn_cast_or_null<DIDerivedType>(T))
        T = DT->getBaseType();
      return dyn_cast_or_null<DICompositeType>(T);
    }
    return nullptr;
  }

  bool isCompleteType(StringRef Name) const {
    const auto *CT = findCompositeType(Name);
    return CT && !CT->isForwardDecl();
  }

  bool isForwardDecl(StringRef Name) const {
    const auto *CT = findCompositeType(Name);
    return CT && CT->isForwardDecl();
  }
};

// Sanity test that structs without a visible constructor definition will emit
// forward declarations of their debug info. The rest of the tests in this file
// rely on this behavior.
TEST_F(StandardLayoutUnionDebugInfoTest, StandaloneStruct) {
  compile(R"cc(
    struct StandaloneSL {
      int x;
      StandaloneSL(int);
    };
    void f(StandaloneSL) {}
  )cc");

  EXPECT_TRUE(isForwardDecl("StandaloneSL"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, NonStandardLayoutStruct) {
  compile(R"cc(
    struct NonSLBase {
      int x;
    };
    struct NonSL : NonSLBase {
      int y;
      NonSL(int);
    };
    void f(NonSL) {}
  )cc");

  EXPECT_TRUE(isForwardDecl("NonSL"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, StandardLayoutUnion) {
  compile(R"cc(
    struct SLInUnion {
      int x;
      SLInUnion(int);
    };

    union SLUnion {
      SLInUnion u;
    };
    void f(SLUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLInUnion"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, RecurseFieldTypes) {
  compile(R"cc(
    struct SLMember {
      int x;
      SLMember(int);
    };

    struct SLInUnion {
      SLMember x;
      SLInUnion(int);
    };

    union SLUnion {
      SLInUnion u;
    };
    void f(SLUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLInUnion"));
  EXPECT_TRUE(isCompleteType("SLMember"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, TemplatedMember) {
  compile(R"cc(
    template <typename T>
    struct TemplatedSL {
      T x;
      TemplatedSL(T);
    };

    union TemplatedUnion {
      TemplatedSL<int> a;
      TemplatedSL<float> b;
    };
    void f(TemplatedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("TemplatedSL<int>"));
  EXPECT_TRUE(isCompleteType("TemplatedSL<float>"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, NonStandardLayoutUnion) {
  compile(R"cc(
    struct NonSLBase {
      int x;
    };
    struct NonSL : NonSLBase {
      int x;
      NonSL(int);
    };

    struct SL {
      int x;
      SL(int);
    };

    union NonSLUnion {
      SL s;
      NonSL n;
    };
    void f(NonSLUnion) {}
  )cc");

  EXPECT_TRUE(isForwardDecl("SL"));
  EXPECT_TRUE(isForwardDecl("NonSL"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, NestedStruct) {
  compile(R"cc(
    union NestedUnion {
      struct NestedSL {
        int a;
        NestedSL(int);
      } n;
    };
    void f(NestedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("NestedSL"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, Array) {
  compile(R"cc(
    struct SLInArray {
      int x;
      SLInArray(int);
    };
    struct SLInMultiArray {
      int y;
      SLInMultiArray(int);
    };
    union ArrayUnion {
      SLInArray arr[3];
      SLInMultiArray multi_arr[2][4];
      int raw;
    };
    void f(ArrayUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLInArray"));
  EXPECT_TRUE(isCompleteType("SLInMultiArray"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, CVQualified) {
  compile(R"cc(
    struct SLConst {
      int x;
      SLConst(int);
    };
    struct SLVolatile {
      int y;
      SLVolatile(int);
    };
    union CVUnion {
      const SLConst c;
      volatile SLVolatile v;
      int raw;
    };
    void f(CVUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLConst"));
  EXPECT_TRUE(isCompleteType("SLVolatile"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, GenericUnionTemplate) {
  compile(R"cc(
    template <typename T>
    union GenericUnion {
      T val;
      int raw;
    };
    struct SLInGenericUnion {
      int x;
      SLInGenericUnion(int);
    };
    void f(GenericUnion<SLInGenericUnion>) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLInGenericUnion"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, AnonymousUnion) {
  compile(R"cc(
    struct SLInAnonUnion {
      int x;
      SLInAnonUnion(int);
    };
    struct EnclosingStruct {
      union {
        SLInAnonUnion a;
        int b;
      };
    };
    void f(EnclosingStruct) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLInAnonUnion"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, Inheritance) {
  compile(R"cc(
    struct EmptyBase {
      EmptyBase(int);
    };
    struct SLDerived : EmptyBase {
      int x;
      SLDerived(int);
    };
    union DerivedUnion {
      SLDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLDerived"));
  EXPECT_TRUE(isCompleteType("EmptyBase"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, TypedefInheritance) {
  compile(R"cc(
    typedef struct EmptyBase {
      EmptyBase(int);
    } EmptyBaseAlias;
    struct SLDerived : EmptyBaseAlias {
      int y;
      SLDerived(int);
    };
    union DerivedUnion {
      SLDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLDerived"));
  EXPECT_TRUE(isCompleteType("EmptyBase"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, MultipleInheritance) {
  compile(R"cc(
    struct EmptyBase1 {
      EmptyBase1(int);
    };
    struct EmptyBase2 {
      EmptyBase2(int);
    };
    struct SLDerived : EmptyBase1, EmptyBase2 {
      int x;
      SLDerived(int);
    };
    union DerivedUnion {
      SLDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("SLDerived"));
  EXPECT_TRUE(isCompleteType("EmptyBase1"));
  EXPECT_TRUE(isCompleteType("EmptyBase2"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, InheritanceNonEmptyBase) {
  compile(R"cc(
    struct SLBase {
      int y;
      SLBase(int);
    };
    struct EmptyDerived : SLBase {
      EmptyDerived(int);
    };
    union DerivedUnion {
      EmptyDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("EmptyDerived"));
  EXPECT_TRUE(isCompleteType("SLBase"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, TypedefInheritanceNonEmptyBase) {
  compile(R"cc(
    typedef struct SLBase {
      int y;
      SLBase(int);
    } SLBaseAlias;
    struct EmptyDerived : SLBaseAlias {
      EmptyDerived(int);
    };
    union DerivedUnion {
      EmptyDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("EmptyDerived"));
  EXPECT_TRUE(isCompleteType("SLBase"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, RecurseInheritance) {
  compile(R"cc(
    struct SLBaseMember {
      int x;
      SLBaseMember(int);
    };
    struct SLBase {
      SLBaseMember y;
      SLBase(int);
    };
    struct EmptyDerived : SLBase {
      EmptyDerived(int);
    };
    union DerivedUnion {
      EmptyDerived d;
      int raw;
    };
    void f(DerivedUnion) {}
  )cc");

  EXPECT_TRUE(isCompleteType("EmptyDerived"));
  EXPECT_TRUE(isCompleteType("SLBase"));
  EXPECT_TRUE(isCompleteType("SLBaseMember"));
}

TEST_F(StandardLayoutUnionDebugInfoTest, IgnorePointerMembers) {
  compile(R"cc(
    struct SLPointerInUnion {
      int x;
      SLPointerInUnion(int);
    };

    union SLUnion {
      SLPointerInUnion* u;
    };
    void f(SLUnion) {}
  )cc");

  EXPECT_TRUE(isForwardDecl("SLPointerInUnion"));
}

} // namespace
