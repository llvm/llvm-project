// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fc++-abi=itanium \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-ITANIUM
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fc++-abi=itanium \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-ITANIUM
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-ARM
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-ARM

// Make sure CXXABI can be overridden with a flag. Check that by excersizing
// pointer to member on arm with itanium cxxabi.

// CIR-ITANIUM: cir.cxx_abi = "itanium"
// No attribute, C++ ABI is derived from the triple.
// CIR-ARM-NOT: cir.cxx_abi = "aarch64"

struct S { virtual void v(); };
void (S::*pv)() = &S::v;

// CIR-ITANIUM: cir.global external @pv = #cir.const_record<{#cir.int<1> : !s64i, #cir.int<0> : !s64i}>
// CIR-ARM: cir.global external @pv = #cir.const_record<{#cir.int<0> : !s64i, #cir.int<1> : !s64i}>

// LLVM-ITANIUM: @pv = {{.*}}{ i64, i64 } { i64 1, i64 0 }
// LLVM-ARM: @pv = {{.*}}{ i64, i64 } { i64 0, i64 1 }
