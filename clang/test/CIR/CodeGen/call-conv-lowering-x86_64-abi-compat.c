// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LINUX --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LINUX --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=3.8 -fclangir -emit-llvm %s -o %t-38-cir.ll
// RUN: FileCheck --check-prefix=LINUX38 --input-file=%t-38-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=3.8 -emit-llvm %s -o %t-38.ll
// RUN: FileCheck --check-prefix=LINUX38 --input-file=%t-38.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=9 -fclangir -emit-llvm %s -o %t-9-cir.ll
// RUN: FileCheck --check-prefix=LINUX9 --input-file=%t-9-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=9 -emit-llvm %s -o %t-9.ll
// RUN: FileCheck --check-prefix=LINUX9 --input-file=%t-9.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=11 -fclangir -emit-llvm %s -o %t-11-cir.ll
// RUN: FileCheck --check-prefix=LINUX11 --input-file=%t-11-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclang-abi-compat=11 -emit-llvm %s -o %t-11.ll
// RUN: FileCheck --check-prefix=LINUX11 --input-file=%t-11.ll %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin -target-feature +avx -fclangir -emit-llvm %s -o %t-darwin-cir.ll
// RUN: FileCheck --check-prefix=DARWIN --input-file=%t-darwin-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -target-feature +avx -emit-llvm %s -o %t-darwin.ll
// RUN: FileCheck --check-prefix=DARWIN --input-file=%t-darwin.ll %s

typedef long long v1ll __attribute__((vector_size(8)));
typedef __int128 v2i128 __attribute__((vector_size(32)));
typedef float v8f __attribute__((vector_size(32)));

// The 0.98 ABI revision sends an eightbyte pair to memory when the high half is
// X87UP and the low half is not X87.  Darwin exempts itself for binary
// compatibility with older GCC, so the same union passes in registers there.
// The int member is what makes the low half INTEGER rather than X87.
typedef union { long double l; int i; } ULongDouble;
void rev98(ULongDouble u) { (void)u; }

// LINUX: define dso_local void @rev98(ptr noundef byval(%union.ULongDouble) align 16 %{{[^,)]+}})
// DARWIN: define void @rev98(i64 %{{[^,)]+}}, double %{{[^,)]+}})

// GCC classifies a 64-bit vector of a 64-bit integer as SSE.  Clang 3.8 and
// older did not, and Darwin, FreeBSD and PlayStation still do not.
void mmx(v1ll v) { (void)v; }

// LINUX: define dso_local void @mmx(double noundef %{{[^,)]+}})
// LINUX38: define dso_local void @mmx(i64 noundef %{{[^,)]+}})
// LINUX9: define dso_local void @mmx(double noundef %{{[^,)]+}})
// DARWIN: define void @mmx(i64 noundef %{{[^,)]+}})

// GCC classifies a vector of __int128 as memory.  Clang 9 and older did not,
// and only Linux and NetBSD follow it.  AVX is on above so that this vector
// would otherwise reach a register, which is what makes the rule observable.
void wide_int128(v2i128 v) { (void)v; }

// LINUX: define dso_local void @wide_int128(ptr noundef byval(<2 x i128>) align 32 %{{[^,)]+}})
// LINUX38: define dso_local void @wide_int128(<2 x i128> noundef %{{[^,)]+}})
// LINUX9: define dso_local void @wide_int128(<2 x i128> noundef %{{[^,)]+}})
// DARWIN: define void @wide_int128(<2 x i128> noundef %{{[^,)]+}})

// A union larger than an eightbyte is classified from the member spanning its
// size, so it reaches registers once the level admits the vector.  Clang 11 and
// older instead treated every member as spanning, which sends this to memory
// because the float member does not.
union UnionWideVector { v8f v; float f; };
void take_union_wide_vector(union UnionWideVector u) { (void)u; }

// LINUX: define dso_local void @take_union_wide_vector(<4 x double> %{{[^,)]+}})
// LINUX11: define dso_local void @take_union_wide_vector(ptr noundef byval(%union.UnionWideVector) align 32 %{{[^,)]+}})
