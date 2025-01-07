// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=REGRET128,MEMRET256,MEMRET512
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -emit-llvm -o - -fclang-abi-compat=19 | FileCheck %s --check-prefixes=REGRET128,REGRET256,REGRET512

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=REGRET128,REGRET256,MEMRET512
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -emit-llvm -o - -fclang-abi-compat=19 | FileCheck %s --check-prefixes=REGRET128,REGRET256,REGRET512

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=REGRET128,REGRET256,REGRET512
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -emit-llvm -o - -fclang-abi-compat=19 | FileCheck %s --check-prefixes=REGRET128,REGRET256,REGRET512

#define __MM_MALLOC_H
#include <x86intrin.h>

// REGRET128: define{{.*}} <4 x float> @get2()
__m128 get2() { __m128 r = (__m128){5,6}; return r; }

// MEMRET256: define{{.*}} void @get4(ptr{{.*}} sret(<8 x float>) align 32 %{{.*}})
// REGRET256: define{{.*}} <8 x float> @get4()
__m256 get4() { __m256 r = (__m256){7,8,9,10}; return r; }

// MEMRET512: define{{.*}} void @get8(ptr{{.*}} sret(<16 x float>) align 64 %{{.*}})
// REGRET512: define{{.*}} <16 x float> @get8()
__m512 get8() { __m512 r = (__m512){7,8,9,10,1,2,3,4}; return r; }