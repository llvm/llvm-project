// __cxa_allocate_exception's thrown_size is size_t: i32 on 32-bit ARM.
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-x86.ll
// RUN: FileCheck --check-prefix=X86 --input-file=%t-x86.ll %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fcxx-exceptions -fexceptions -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t-ogcg.ll %s

void f() { throw 42; }

// The width is resolved during lowering to LLVM.
// CIR-LABEL: cir.func{{.*}} @_Z1fv()
// CIR: cir.alloc.exception 4

// CIR emits the declare first, classic CodeGen emits the call first.
// ARM-DAG: declare ptr @__cxa_allocate_exception(i32)
// ARM-DAG: call ptr @__cxa_allocate_exception(i32 4)

// X86: declare ptr @__cxa_allocate_exception(i64)
// X86: call ptr @__cxa_allocate_exception(i64 4)
