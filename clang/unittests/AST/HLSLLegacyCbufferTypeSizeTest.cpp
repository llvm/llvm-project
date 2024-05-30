//===- unittests/AST/HLSLLegacyCbufferTypeSizeTest.cpp -- cbuf size test --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for HLSLBufferDecl::calculateLegacyCbufferSize.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

// FIXME: Let buildASTFromCodeWithArgs find hlsl.h instead of using "nostdinc".

TEST(HLSLLegacyCbufferTypeSize, BasicTypes) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {
    // built-in scalar data types:

    #ifdef __HLSL_ENABLE_16_BIT
    // 16-bit integer.
    typedef short int16_t;
    #endif

    // 64-bit integer.
    typedef long int64_t;

    } // namespace hlsl

    struct A {
      float a;
      double b;
      float  c;
      half   d;
      int16_t e;
      int64_t f;
      int     g;
    };
  )hlsl";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code,
                                        /*Args=*/
                                        {"--driver-mode=dxc", "-T", "lib_6_3",
                                         "-enable-16bit-types", "-nostdinc"},
                                        /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);

  //  struct A
  //  {

  //      float a;                                      ; Offset:    0
  //      double b;                                     ; Offset:    8
  //      float c;                                      ; Offset:   16
  //      half d;                                       ; Offset:   20
  //      int16_t e;                                    ; Offset:   22
  //      int64_t f;                                    ; Offset:   24
  //      int g;                                        ; Offset:   32

  //  } A;                                              ; Offset:    0 Size: 36
  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 36u * 8u);
}

TEST(HLSLLegacyCbufferTypeSize, VectorTypes) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {
    // built-in scalar data types:

    #ifdef __HLSL_ENABLE_16_BIT
    // 16-bit integer.
    typedef short int16_t;
    #endif

    // 64-bit integer.
    typedef long int64_t;

    // built-in vector data types:

    typedef vector<half, 2> half2;
    typedef vector<half, 3> half3;
    typedef vector<half, 4> half4;

    typedef vector<float, 2> float2;
    typedef vector<float, 3> float3;
    typedef vector<float, 4> float4;
    typedef vector<double, 2> double2;
    typedef vector<double, 3> double3;
    typedef vector<double, 4> double4;

    } // namespace hlsl

    struct A {
      double  B0;
      float3  B1;
      float   B2;
      double3 B3;
      half    B4;
      double2 B5;
      float   B6;
      half3   B7;
      half3   B8;
    };
  )hlsl";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code,
                                        /*Args=*/
                                        {"--driver-mode=dxc", "-T", "lib_6_3",
                                         "-enable-16bit-types", "-nostdinc"},
                                        /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);
  //  struct A
  //  {

  //      double B0;                                    ; Offset:    0
  //      float3 B1;                                    ; Offset:   16
  //      float B2;                                     ; Offset:   28
  //      double3 B3;                                   ; Offset:   32
  //      half B4;                                      ; Offset:   56
  //      double2 B5;                                   ; Offset:   64
  //      float B6;                                     ; Offset:   80
  //      half3 B7;                                     ; Offset:   84
  //      half3 B8;                                     ; Offset:   90

  //  } A;                                              ; Offset:    0 Size: 96
  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 96u * 8u);
}

TEST(HLSLLegacyCbufferTypeSize, ArrayTypes) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {
    // built-in scalar data types:

    #ifdef __HLSL_ENABLE_16_BIT
    // 16-bit integer.
    typedef short int16_t;
    #endif

    // 64-bit integer.
    typedef long int64_t;

    // built-in vector data types:

    typedef vector<half, 2> half2;
    typedef vector<half, 3> half3;
    typedef vector<half, 4> half4;

    typedef vector<float, 2> float2;
    typedef vector<float, 3> float3;
    typedef vector<float, 4> float4;
    typedef vector<double, 2> double2;
    typedef vector<double, 3> double3;
    typedef vector<double, 4> double4;

    } // namespace hlsl

    struct A {
      double C0[2];
      float3 C1[3];
      float  C2;
      double C3[3];
      half   C4;
      double2 C5[1];
      float  C6;
      half3  C7[2];
      half3  C8;
    };
  )hlsl";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code,
                                        /*Args=*/
                                        {"--driver-mode=dxc", "-T", "lib_6_3",
                                         "-enable-16bit-types", "-nostdinc"},
                                        /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);

  //  struct A
  //  {

  //      double C0[2];                                 ; Offset:    0
  //      float3 C1[3];                                 ; Offset:   32
  //      float C2;                                     ; Offset:   76
  //      double C3[3];                                 ; Offset:   80
  //      half C4;                                      ; Offset:  120
  //      double2 C5[1];                                ; Offset:  128
  //      float C6;                                     ; Offset:  144
  //      half3 C7[2];                                  ; Offset:  160
  //      half3 C8;                                     ; Offset:  182

  //  } A;                                              ; Offset:    0 Size: 188
  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 188u * 8u);
}

TEST(HLSLLegacyCbufferTypeSize, Vec3) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {
    // built-in scalar data types:

    #ifdef __HLSL_ENABLE_16_BIT
    // 16-bit integer.
    typedef short int16_t;
    #endif

    // 64-bit integer.
    typedef long int64_t;

    // built-in vector data types:

    typedef vector<half, 2> half2;
    typedef vector<half, 3> half3;
    typedef vector<half, 4> half4;

    typedef vector<float, 2> float2;
    typedef vector<float, 3> float3;
    typedef vector<float, 4> float4;
    typedef vector<double, 2> double2;
    typedef vector<double, 3> double3;
    typedef vector<double, 4> double4;

    } // namespace hlsl

    struct A {
      double3 D9[3];
      half3 D10;
    };
  )hlsl";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code,
                                        /*Args=*/
                                        {"--driver-mode=dxc", "-T", "lib_6_3",
                                         "-enable-16bit-types", "-nostdinc"},
                                        /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);

  //  struct A
  //  {
  //
  //      double3 D9[3];                                ; Offset:    0
  //      half3 D10;                                    ; Offset:   88
  //
  //  } A;                                              ; Offset:    0 Size: 94
  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 94u * 8u);
}

TEST(HLSLLegacyCbufferTypeSize, NestedStruct) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {
    // built-in scalar data types:

    #ifdef __HLSL_ENABLE_16_BIT
    // 16-bit integer.
    typedef short int16_t;
    #endif

    // 64-bit integer.
    typedef long int64_t;

    // built-in vector data types:

    typedef vector<half, 2> half2;
    typedef vector<half, 3> half3;
    typedef vector<half, 4> half4;

    typedef vector<float, 2> float2;
    typedef vector<float, 3> float3;
    typedef vector<float, 4> float4;
    typedef vector<double, 2> double2;
    typedef vector<double, 3> double3;
    typedef vector<double, 4> double4;

    } // namespace hlsl

    struct S0
    {
        double B0;
        float3 B1;
        float B2;
        double3 B3;
        half B4;
        double2 B5;
        float B6;
        half3 B7;
        half3 B8;
    };

    struct S1
    {
        float A0;
        double A1;
        float A2;
        half A3;
        int16_t A4;
        int64_t A5;
        int A6;
    };

    struct S2
    {
        double B0;
        float3 B1;
        float B2;
        double3 B3;
        half B4;
        double2 B5;
        float B6;
        half3 B7;
        half3 B8;
    };

    struct S3
    {
        S1 C0;
        float C1[1];
        S2 C2[2];
        half C3;
    };           

    struct A {
      int E0;
      S0  E1;
      half E2;
      S3 E3;
      double E4;
    };
  )hlsl";

  auto AST =
      tooling::buildASTFromCodeWithArgs(Code,
                                        /*Args=*/
                                        {"--driver-mode=dxc", "-T", "lib_6_3",
                                         "-enable-16bit-types", "-nostdinc"},
                                        /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);

  //  struct E
  //  {
  //      int E0;                                       ; Offset:    0
  //      struct struct.S0
  //      {

  //          double B0;                                ; Offset:   16
  //          float3 B1;                                ; Offset:   32
  //          float B2;                                 ; Offset:   44
  //          double3 B3;                               ; Offset:   48
  //          half B4;                                  ; Offset:   72
  //          double2 B5;                               ; Offset:   80
  //          float B6;                                 ; Offset:   96
  //          half3 B7;                                 ; Offset:  100
  //          half3 B8;                                 ; Offset:  106

  //      } E1;                                         ; Offset:   16

  //      half E2;                                      ; Offset:  112
  //      struct struct.S3
  //      {

  //          struct struct.S1
  //          {

  //              float A0;                             ; Offset:  128
  //              double A1;                            ; Offset:  136
  //              float A2;                             ; Offset:  144
  //              half A3;                              ; Offset:  148
  //              int16_t A4;                           ; Offset:  150
  //              int64_t A5;                           ; Offset:  152
  //              int A6;                               ; Offset:  160

  //          } C0;                                     ; Offset:  128

  //          float C1[1];                              ; Offset:  176
  //          struct struct.S2
  //          {

  //              double B0;                            ; Offset:  192
  //              float3 B1;                            ; Offset:  208
  //              float B2;                             ; Offset:  220
  //              double3 B3;                           ; Offset:  224
  //              half B4;                              ; Offset:  248
  //              double2 B5;                           ; Offset:  256
  //              float B6;                             ; Offset:  272
  //              half3 B7;                             ; Offset:  276
  //              half3 B8;                             ; Offset:  282

  //          } C2[2];;                                 ; Offset:  192

  //          half C3;                                  ; Offset:  384

  //      } E3;                                         ; Offset:  128

  //      double E4;                                    ; Offset:  392

  //  } E;                                              ; Offset:    0 Size: 400

  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 400u * 8u);
}

TEST(HLSLLegacyCbufferTypeSize, NestedStructDisable16bitTypes) {
  constexpr char Code[] = R"hlsl(
    namespace hlsl {

    // built-in vector data types:

    typedef vector<half, 2> half2;
    typedef vector<half, 3> half3;
    typedef vector<half, 4> half4;

    typedef vector<float, 2> float2;
    typedef vector<float, 3> float3;
    typedef vector<float, 4> float4;
    typedef vector<double, 2> double2;
    typedef vector<double, 3> double3;
    typedef vector<double, 4> double4;

    } // namespace hlsl

    struct S0
    {
        double B0;
        float3 B1;
        float B2;
        double3 B3;
        half B4;
        double2 B5;
        float B6;
        half3 B7;
        half3 B8;
    };

    struct S1
    {
        float A0;
        double A1;
        float A2;
        half A3;
        half A4;
        double A5;
        int A6;
    };

    struct S2
    {
        double B0;
        float3 B1;
        float B2;
        double3 B3;
        half B4;
        double2 B5;
        float B6;
        half3 B7;
        half3 B8;
    };

    struct S3
    {
        S1 C0;
        float C1[1];
        S2 C2[2];
        half C3;
    };           

    struct A {
      int E0;
      S0  E1;
      half E2;
      S3 E3;
      double E4;
    };
  )hlsl";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code,
      /*Args=*/{"--driver-mode=dxc", "-T", "lib_6_3", "-nostdinc"},
      /* FileName*/ "input.hlsl");
  ASTContext &Ctx = AST->getASTContext();

  auto Results = match(decl(cxxRecordDecl(hasName("A")).bind("A")), Ctx);

  //  struct E
  //  {

  //      int E0;                                       ; Offset:    0
  //      struct struct.S0
  //      {

  //          double B0;                                ; Offset:   16
  //          float3 B1;                                ; Offset:   32
  //          float B2;                                 ; Offset:   44
  //          double3 B3;                               ; Offset:   48
  //          float B4;                                 ; Offset:   72
  //          double2 B5;                               ; Offset:   80
  //          float B6;                                 ; Offset:   96
  //          float3 B7;                                ; Offset:  100
  //          float3 B8;                                ; Offset:  112

  //      } E1;                                         ; Offset:   16

  //      float E2;                                     ; Offset:  124
  //      struct struct.S3
  //      {

  //          struct struct.S1
  //          {

  //              float A0;                             ; Offset:  128
  //              double A1;                            ; Offset:  136
  //              float A2;                             ; Offset:  144
  //              float A3;                             ; Offset:  148
  //              float A4;                             ; Offset:  152
  //              double A5;                            ; Offset:  160
  //              int A6;                               ; Offset:  168

  //          } C0;                                     ; Offset:  128

  //          float C1[1];                              ; Offset:  176
  //          struct struct.S2
  //          {

  //              double B0;                            ; Offset:  192
  //              float3 B1;                            ; Offset:  208
  //              float B2;                             ; Offset:  220
  //              double3 B3;                           ; Offset:  224
  //              float B4;                             ; Offset:  248
  //              double2 B5;                           ; Offset:  256
  //              float B6;                             ; Offset:  272
  //              float3 B7;                            ; Offset:  276
  //              float3 B8;                            ; Offset:  288

  //          } C2[2];;                                 ; Offset:  192

  //          float C3;                                 ; Offset:  412

  //      } E3;                                         ; Offset:  128

  //      double E4;                                    ; Offset:  416

  //  } E;                                              ; Offset:    0 Size: 424

  auto const *A = selectFirst<RecordDecl>("A", Results);
  unsigned SizeA = HLSLBufferDecl::calculateLegacyCbufferSize(
      Ctx, QualType(A->getTypeForDecl(), 0));
  ASSERT_EQ(SizeA, 424u * 8u);
}
