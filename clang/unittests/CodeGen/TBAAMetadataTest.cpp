//=== unittests/CodeGen/TBAAMetadataTest.cpp - Checks metadata generation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRMatchers.h"
#include "TestCompiler.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

struct TBAATestCompiler : public TestCompiler {
  TBAATestCompiler(clang::LangOptions LO, clang::CodeGenOptions CGO)
    : TestCompiler(LO, CGO) {}
  static clang::CodeGenOptions getCommonCodeGenOpts() {
    clang::CodeGenOptions CGOpts;
    CGOpts.StructPathTBAA = 1;
    CGOpts.OptimizationLevel = 1;
    return CGOpts;
  }
};

auto OmnipotentCharC = MMTuple(
  MMString("omnipotent char"),
  MMTuple(
    MMString("Simple C/C++ TBAA")),
  MConstInt(0, 64)
);


auto OmnipotentCharCXX = MMTuple(
  MMString("omnipotent char"),
  MMTuple(
    MMString("Simple C++ TBAA")),
  MConstInt(0, 64)
);


TEST(TBAAMetadataTest, BasicTypes) {
  const char TestProgram[] = R"**(
    void func(char *CP, short *SP, int *IP, long long *LP, void **VPP,
              int **IPP) {
      *CP = 4;
      *SP = 11;
      *IP = 601;
      *LP = 604;
      *VPP = CP;
      *IPP = IP;
    }
  )**";

  clang::LangOptions LO;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 8),
        MMTuple(
          OmnipotentCharC,
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(601, 32),
        MMTuple(
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(604, 64),
        MMTuple(
          MMTuple(
            MMString("long long"),
            OmnipotentCharC,
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MValType(Type::getInt8PtrTy(Compiler.Context)),
        MMTuple(
          MMTuple(
            MMString("any pointer"),
            OmnipotentCharC,
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MValType(Type::getInt32PtrTy(Compiler.Context)),
        MMTuple(
          MMTuple(
            MMString("any pointer"),
            OmnipotentCharC,
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, CFields) {
  const char TestProgram[] = R"**(
    struct ABC {
       short f16;
       int f32;
       long long f64;
       unsigned short f16_2;
       unsigned f32_2;
       unsigned long long f64_2;
    };

    void func(struct ABC *A) {
      A->f32 = 4;
      A->f16 = 11;
      A->f64 = 601;
      A->f16_2 = 22;
      A->f32_2 = 77;
      A->f64_2 = 604;
    }
  )**";

  clang::LangOptions LO;
  LO.C11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto StructABC = MMTuple(
    MMString("ABC"),
    MMTuple(
      MMString("short"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(4),
    MMTuple(
      MMString("long long"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(8),
    MSameAs(1),
    MConstInt(16),
    MSameAs(3),
    MConstInt(20),
    MSameAs(5),
    MConstInt(24));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(601, 64),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("long long"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(8))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(22, 16),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(16))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(20))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(604, 64),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("long long"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(24))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, CTypedefFields) {
  const char TestProgram[] = R"**(
    typedef struct {
       short f16;
       int f32;
    } ABC;
    typedef struct {
       short value_f16;
       int value_f32;
    } CDE;

    void func(ABC *A, CDE *B) {
      A->f32 = 4;
      A->f16 = 11;
      B->value_f32 = 44;
      B->value_f16 = 111;
    }
  )**";

  clang::LangOptions LO;
  LO.C11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto NamelessStruct = MMTuple(
    MMString(""),
    MMTuple(
      MMString("short"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(4));

  const Metadata *MetaABC = nullptr;
  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          MMSave(MetaABC, NamelessStruct),
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          NamelessStruct,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  const Metadata *MetaCDE = nullptr;
  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(44, 32),
        MMTuple(
          MMSave(MetaCDE, NamelessStruct),
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(111, 16),
        MMTuple(
          NamelessStruct,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  // FIXME: Nameless structures used in definitions of 'ABC' and 'CDE' are
  // different structures and must be described by different descriptors.
  //ASSERT_TRUE(MetaABC != MetaCDE);
}

TEST(TBAAMetadataTest, CTypedefFields2) {
  const char TestProgram[] = R"**(
    typedef struct {
       short f16;
       int f32;
    } ABC;
    typedef struct {
       short f16;
       int f32;
    } CDE;

    void func(ABC *A, CDE *B) {
      A->f32 = 4;
      A->f16 = 11;
      B->f32 = 44;
      B->f16 = 111;
    }
  )**";

  clang::LangOptions LO;
  LO.C11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto NamelessStruct = MMTuple(
    MMString(""),
    MMTuple(
      MMString("short"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(4));

  const Metadata *MetaABC = nullptr;
  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          MMSave(MetaABC, NamelessStruct),
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          NamelessStruct,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  const Metadata *MetaCDE = nullptr;
  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(44, 32),
        MMTuple(
          MMSave(MetaCDE, NamelessStruct),
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(111, 16),
        MMTuple(
          NamelessStruct,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  // FIXME: Nameless structures used in definitions of 'ABC' and 'CDE' are
  // different structures, although they have the same field sequence. They must
  // be described by different descriptors.
  //ASSERT_TRUE(MetaABC != MetaCDE);
}

TEST(TBAAMetadataTest, CTypedefFields3) {
  const char TestProgram[] = R"**(
    typedef struct {
       short f16;
       int f32;
    } ABC;
    typedef struct {
       int f32;
       short f16;
    } CDE;

    void func(ABC *A, CDE *B) {
      A->f32 = 4;
      A->f16 = 11;
      B->f32 = 44;
      B->f16 = 111;
    }
  )**";

  clang::LangOptions LO;
  LO.C11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto NamelessStruct1 = MMTuple(
    MMString(""),
    MMTuple(
      MMString("short"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(4));

  auto NamelessStruct2 = MMTuple(
    MMString(""),
    MMTuple(
      MMString("int"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("short"),
      OmnipotentCharC,
      MConstInt(0)),
    MConstInt(4));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          NamelessStruct1,
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          NamelessStruct1,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(44, 32),
        MMTuple(
          NamelessStruct2,
          MMTuple(
            MMString("int"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(111, 16),
        MMTuple(
          NamelessStruct2,
          MMTuple(
            MMString("short"),
            OmnipotentCharC,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, CXXFields) {
  const char TestProgram[] = R"**(
    struct ABC {
       short f16;
       int f32;
       long long f64;
       unsigned short f16_2;
       unsigned f32_2;
       unsigned long long f64_2;
    };

    void func(struct ABC *A) {
      A->f32 = 4;
      A->f16 = 11;
      A->f64 = 601;
      A->f16_2 = 22;
      A->f32_2 = 77;
      A->f64_2 = 604;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto StructABC = MMTuple(
    MMString("_ZTS3ABC"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(4),
    MMTuple(
      MMString("long long"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(8),
    MSameAs(1),
    MConstInt(16),
    MSameAs(3),
    MConstInt(20),
    MSameAs(5),
    MConstInt(24));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(601, 64),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("long long"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(8))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(22, 16),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(16))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(20))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(604, 64),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("long long"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(24))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, CXXTypedefFields) {
  const char TestProgram[] = R"**(
    typedef struct {
       short f16;
       int f32;
    } ABC;
    typedef struct {
       short value_f16;
       int value_f32;
    } CDE;

    void func(ABC *A, CDE *B) {
      A->f32 = 4;
      A->f16 = 11;
      B->value_f32 = 44;
      B->value_f16 = 111;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto StructABC = MMTuple(
    MMString("_ZTS3ABC"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(4));

  auto StructCDE = MMTuple(
    MMString("_ZTS3CDE"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(4));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(4, 32),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(11, 16),
        MMTuple(
          StructABC,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(44, 32),
        MMTuple(
          StructCDE,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(111, 16),
        MMTuple(
          StructCDE,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, StructureFields) {
  const char TestProgram[] = R"**(
    struct Inner {
      int f32;
    };

    struct Outer {
      short f16;
      Inner b1;
      Inner b2;
    };

    void func(Outer *S) {
      S->f16 = 14;
      S->b1.f32 = 35;
      S->b2.f32 = 77;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto StructInner = MMTuple(
    MMString("_ZTS5Inner"),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0));

  auto StructOuter = MMTuple(
    MMString("_ZTS5Outer"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    StructInner,
    MConstInt(4),
    MSameAs(3),
    MConstInt(8));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(14, 16),
        MMTuple(
          StructOuter,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(35, 32),
        MMTuple(
          StructOuter,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          StructOuter,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(8))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, ArrayFields) {
  const char TestProgram[] = R"**(
    struct Inner {
      int f32;
    };

    struct Outer {
      short f16;
      Inner b1[2];
    };

    void func(Outer *S) {
      S->f16 = 14;
      S->b1[0].f32 = 35;
      S->b1[1].f32 = 77;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto StructInner = MMTuple(
    MMString("_ZTS5Inner"),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0));

  auto StructOuter = MMTuple(
    MMString("_ZTS5Outer"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    OmnipotentCharCXX,    // FIXME: Info about array field is lost.
    MConstInt(4));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(14, 16),
        MMTuple(
          StructOuter,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(35, 32),
        MMTuple(
          StructInner,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          StructInner,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, BaseClass) {
  const char TestProgram[] = R"**(
    struct Base {
      int f32;
    };

    struct Derived : public Base {
      short f16;
    };

    void func(Base *B, Derived *D) {
      B->f32 = 14;
      D->f16 = 35;
      D->f32 = 77;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto ClassBase = MMTuple(
    MMString("_ZTS4Base"),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0));

  auto ClassDerived =
      MMTuple(MMString("_ZTS7Derived"), ClassBase, MConstInt(0),
              MMTuple(MMString("short"), OmnipotentCharCXX, MConstInt(0)),
              MConstInt(4));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(14, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(35, 16),
        MMTuple(
          ClassDerived,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, PolymorphicClass) {
  const char TestProgram[] = R"**(
    struct Base {
      virtual void m1(int *) = 0;
      int f32;
    };

    struct Derived : public Base {
      virtual void m1(int *) override;
      short f16;
    };

    void func(Base *B, Derived *D) {
      B->f32 = 14;
      D->f16 = 35;
      D->f32 = 77;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto ClassBase = MMTuple(
    MMString("_ZTS4Base"),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(Compiler.PtrSize));

  auto ClassDerived =
      MMTuple(MMString("_ZTS7Derived"), ClassBase, MConstInt(0),
              MMTuple(MMString("short"), OmnipotentCharCXX, MConstInt(0)),
              MConstInt(Compiler.PtrSize + 4));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(14, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(Compiler.PtrSize))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(35, 16),
        MMTuple(
          ClassDerived,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(Compiler.PtrSize + 4))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(Compiler.PtrSize))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, VirtualBase) {
  const char TestProgram[] = R"**(
    struct Base {
      int f32;
    };

    struct Derived : public virtual Base {
      short f16;
    };

    void func(Base *B, Derived *D) {
      B->f32 = 14;
      D->f16 = 35;
      D->f32 = 77;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto ClassBase = MMTuple(
    MMString("_ZTS4Base"),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0));

  auto ClassDerived = MMTuple(
    MMString("_ZTS7Derived"),
    MMTuple(
      MMString("short"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(Compiler.PtrSize));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MConstInt(14, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(35, 16),
        MMTuple(
          ClassDerived,
          MMTuple(
            MMString("short"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(Compiler.PtrSize))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Load,
        MMTuple(
          MMTuple(
            MMString("vtable pointer"),
            MMTuple(
              MMString("Simple C++ TBAA")),
            MConstInt(0)),
          MSameAs(0),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(77, 32),
        MMTuple(
          ClassBase,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);
}

TEST(TBAAMetadataTest, TemplSpec) {
  const char TestProgram[] = R"**(
    template<typename T1, typename T2>
    struct ABC {
      T1 f1;
      T2 f2;
    };

    void func(ABC<double, int> *p) {
      p->f1 = 12.1;
      p->f2 = 44;
    }
  )**";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TBAATestCompiler Compiler(LO, TBAATestCompiler::getCommonCodeGenOpts());
  Compiler.init(TestProgram);
  const BasicBlock *BB = Compiler.compile();

  auto SpecABC = MMTuple(
    MMString("_ZTS3ABCIdiE"),
    MMTuple(
      MMString("double"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(0),
    MMTuple(
      MMString("int"),
      OmnipotentCharCXX,
      MConstInt(0)),
    MConstInt(8));

  const Instruction *I = match(BB,
      MInstruction(Instruction::Store,
        MValType(MType([](const Type &T)->bool { return T.isDoubleTy(); })),
        MMTuple(
          SpecABC,
          MMTuple(
            MMString("double"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(0))));
  ASSERT_TRUE(I);

  I = matchNext(I,
      MInstruction(Instruction::Store,
        MConstInt(44, 32),
        MMTuple(
          SpecABC,
          MMTuple(
            MMString("int"),
            OmnipotentCharCXX,
            MConstInt(0)),
          MConstInt(8))));
  ASSERT_TRUE(I);
}
}
