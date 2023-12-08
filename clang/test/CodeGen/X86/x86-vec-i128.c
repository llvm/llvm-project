// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,MEM256ALIGN32,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +sse2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,MEM256ALIGN32,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +sse2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEM256ALIGN16,MEM512ALIGN16
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +sse2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEM256ALIGN32,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +sse2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEM256ALIGN32,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -S -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,MEM256ALIGN32,MEM512ALIGN64

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +avx -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +avx -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEM512ALIGN32
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +avx -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +avx -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEM512ALIGN64
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -S -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEM512ALIGN64

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,CLANG10ABI512
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +avx512f -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,CLANG10ABI512
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +avx512f -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +avx512f -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +avx512f -S -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -S -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512

typedef unsigned long long v16u64 __attribute__((vector_size(16)));
typedef unsigned __int128 v16u128 __attribute__((vector_size(16)));

v16u64 test_v16u128(v16u64 a, v16u128 b) {
// CLANG10ABI128: define{{.*}} <2 x i64> @test_v16u128(<2 x i64> noundef %{{.*}}, <2 x i64> noundef %{{.*}})
// CLANG9ABI128: define{{.*}} <2 x i64> @test_v16u128(<2 x i64> noundef %{{.*}}, <1 x i128> noundef %{{.*}})
  return a + (v16u64)b;
}

typedef unsigned long long v32u64 __attribute__((vector_size(32)));
typedef unsigned __int128 v32u128 __attribute__((vector_size(32)));

v32u64 test_v32u128(v32u64 a, v32u128 b) {
// MEM256ALIGN16: define{{.*}} <4 x i64> @test_v32u128(ptr noundef byval(<4 x i64>) align 16 %{{.*}}, ptr noundef byval(<2 x i128>) align 16 %{{.*}})
// MEM256ALIGN32: define{{.*}} <4 x i64> @test_v32u128(ptr noundef byval(<4 x i64>) align 32 %{{.*}}, ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// CLANG10ABI256: define{{.*}} <4 x i64> @test_v32u128(<4 x i64> noundef %{{.*}}, ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// CLANG9ABI256: define{{.*}} <4 x i64> @test_v32u128(<4 x i64> noundef %{{.*}}, <2 x i128> noundef %{{.*}})
  return a + (v32u64)b;
}

typedef unsigned long long v64u64 __attribute__((vector_size(64)));
typedef unsigned __int128 v64u128 __attribute__((vector_size(64)));

v64u64 test_v64u128(v64u64 a, v64u128 b) {
// MEM512ALIGN16: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 16 %{{.*}}, ptr noundef byval(<4 x i128>) align 16 %{{.*}})
// MEM512ALIGN32: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 32 %{{.*}}, ptr noundef byval(<4 x i128>) align 32 %{{.*}})
// MEM512ALIGN64: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 64 %{{.*}}, ptr noundef byval(<4 x i128>) align 64 %{{.*}})
// CLANG10ABI512: define{{.*}} <8 x i64> @test_v64u128(<8 x i64> noundef %{{.*}}, ptr noundef byval(<4 x i128>) align 64 %{{.*}})
// CLANG9ABI512: define{{.*}} <8 x i64> @test_v64u128(<8 x i64> noundef %{{.*}}, <4 x i128> noundef %{{.*}})
  return a + (v64u64)b;
}
