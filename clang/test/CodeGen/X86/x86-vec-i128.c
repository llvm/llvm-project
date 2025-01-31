// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,MEMRETMEMARG256ALIGN32,MEMRETMEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,MEMRETMEMARG256ALIGN32,MEMRETMEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEMARG256ALIGN16,MEMARG512ALIGN16
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEMARG256ALIGN32,MEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +sse2 -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,MEMARG256ALIGN32,MEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +sse2 -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,MEMARG256ALIGN32,MEMARG512ALIGN64

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,MEMRETMEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,MEMRETMEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEMARG512ALIGN32
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +avx -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEMARG512ALIGN64
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,MEMARG512ALIGN64

// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,CLANG10ABI512
// RUN: %clang_cc1 -triple x86_64-netbsd %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG10ABI128,CLANG10ABI256,CLANG10ABI512
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-scei-ps4 %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 %s -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -target-feature +avx512f -emit-llvm -o - -fclang-abi-compat=9 | FileCheck %s --check-prefixes=CLANG9ABI128,CLANG9ABI256,CLANG9ABI512

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
// MEMARG256ALIGN16: define{{.*}} <4 x i64> @test_v32u128(ptr noundef byval(<4 x i64>) align 16 %{{.*}}, ptr noundef byval(<2 x i128>) align 16 %{{.*}})
// MEMARG256ALIGN32: define{{.*}} <4 x i64> @test_v32u128(ptr noundef byval(<4 x i64>) align 32 %{{.*}}, ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// MEMRETMEMARG256ALIGN32: define{{.*}} void @test_v32u128(ptr{{.*}} sret(<4 x i64>) align 32 %{{.*}}, ptr noundef byval(<4 x i64>) align 32 %{{.*}}, ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// CLANG10ABI256: define{{.*}} <4 x i64> @test_v32u128(<4 x i64> noundef %{{.*}}, ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// CLANG9ABI256: define{{.*}} <4 x i64> @test_v32u128(<4 x i64> noundef %{{.*}}, <2 x i128> noundef %{{.*}})
  return a + (v32u64)b;
}

typedef unsigned long long v64u64 __attribute__((vector_size(64)));
typedef unsigned __int128 v64u128 __attribute__((vector_size(64)));

v64u64 test_v64u128(v64u64 a, v64u128 b) {
// MEMARG512ALIGN16: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 16 %{{.*}}, ptr noundef byval(<4 x i128>) align 16 %{{.*}})
// MEMARG512ALIGN32: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 32 %{{.*}}, ptr noundef byval(<4 x i128>) align 32 %{{.*}})
// MEMARG512ALIGN64: define{{.*}} <8 x i64> @test_v64u128(ptr noundef byval(<8 x i64>) align 64 %{{.*}}, ptr noundef byval(<4 x i128>) align 64 %{{.*}})
// MEMRETMEMARG512ALIGN64: define{{.*}} void @test_v64u128(ptr{{.*}} sret(<8 x i64>) align 64 %{{.*}}, ptr noundef byval(<8 x i64>) align 64 %{{.*}}, ptr noundef byval(<4 x i128>) align 64 %{{.*}})
// CLANG10ABI512: define{{.*}} <8 x i64> @test_v64u128(<8 x i64> noundef %{{.*}}, ptr noundef byval(<4 x i128>) align 64 %{{.*}})
// CLANG9ABI512: define{{.*}} <8 x i64> @test_v64u128(<8 x i64> noundef %{{.*}}, <4 x i128> noundef %{{.*}})
  return a + (v64u64)b;
}
