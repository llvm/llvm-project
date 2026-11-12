// RUN: %clang_cc1 -std=c++20 -triple i686-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple i686-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple i686-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// The result of sizeof/alignof/__builtin_vectorelements is size_t, which is
// 32 bits wide on this target. The emitted constants must use that width
// rather than a hardcoded 64-bit type.

using size_t = decltype(sizeof(int));

size_t size_of_int() { return sizeof(int); }

// CIR-LABEL: cir.func {{.*}}@_Z11size_of_intv() -> {{.*}}!u32i
// CIR:         cir.const #cir.int<4> : !u32i

// LLVM-LABEL: define {{.*}}i32 @_Z11size_of_intv()
// LLVM:         {{.*}}i32 4

size_t align_of_double() { return alignof(double); }

// CIR-LABEL: cir.func {{.*}}@_Z15align_of_doublev() -> {{.*}}!u32i
// CIR:         cir.const #cir.int<4> : !u32i

// LLVM-LABEL: define {{.*}}i32 @_Z15align_of_doublev()
// LLVM:         {{.*}}i32 4

typedef int vi4 __attribute__((vector_size(16)));

size_t vector_elements(vi4 v) { return __builtin_vectorelements(v); }

// CIR-LABEL: cir.func {{.*}}@_Z15vector_elementsDv4_i({{.*}}) -> {{.*}}!u32i
// CIR:         cir.const #cir.int<4> : !u32i

// LLVM-LABEL: define {{.*}}i32 @_Z15vector_elementsDv4_i
// LLVM:         {{.*}}i32 4
