// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__m256i test__builtin_ia32_permdi256()
{
    // CIR-LABEL: test__builtin_ia32_permdi256
    // CIR: cir.vec.shuffle({{%.*}}, {{%.*}}: !cir.vector<4 x !s64i>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !s64i>
    // LLVM-LABEL: test__builtin_ia32_permdi256
    // LLVM: shufflevector <4 x i64> {{%.*}}, <4 x i64> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
    // OGCG-LABEL: test__builtin_ia32_permdi256
    // OGCG: shufflevector <4 x i64> {{%.*}}, <4 x i64> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
    __v4di vec = {0, 1, 2, 3};
    return __builtin_ia32_permdi256(vec, 1);
} 

__m512i test__builtin_ia32_permdi512()
{
    // CIR-LABEL: test__builtin_ia32_permdi512
    // CIR: cir.vec.shuffle({{%.*}}, {{%.*}}: !cir.vector<8 x !s64i>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<8 x !s64i>
    // LLVM-LABEL: test__builtin_ia32_permdi512
    // LLVM: shufflevector <8 x i64> {{%.*}}, <8 x i64> poison, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
    // OGCG-LABEL: test__builtin_ia32_permdi512
    // OGCG: shufflevector <8 x i64> {{%.*}}, <8 x i64> poison, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
    __v8di vec = {0, 1, 2, 3, 4, 5, 6, 7};
    return __builtin_ia32_permdi512(vec, 1);
}

__m256d test__builtin_ia32_permdf256()
{
    // CIR-LABEL: test__builtin_ia32_permdf256
    // CIR: cir.vec.shuffle({{%.*}}, {{%.*}}: !cir.vector<4 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !cir.double>
    // LLVM-LABEL: test__builtin_ia32_permdf256
    // LLVM: shufflevector <4 x double> {{%.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
    // OGCG-LABEL: test__builtin_ia32_permdf256
    // OGCG: shufflevector <4 x double> {{%.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
    __v4df vec = {0, 1, 2, 3};
    return __builtin_ia32_permdf256(vec, 1);
}   

__m512d test__builtin_ia32_permdf512()
{
    // CIR-LABEL: test__builtin_ia32_permdf512
    // CIR: cir.vec.shuffle({{%.*}}, {{%.*}}: !cir.vector<8 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<8 x !cir.double>
    // LLVM-LABEL: test__builtin_ia32_permdf512
    // LLVM: shufflevector <8 x double> {{%.*}}, <8 x double> poison, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
    // OGCG-LABEL: test__builtin_ia32_permdf512
    // OGCG: shufflevector <8 x double> {{%.*}}, <8 x double> poison, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
    __v8df vec = {0, 1, 2, 3, 4, 5, 6, 7};
    return __builtin_ia32_permdf512(vec, 1);
}
