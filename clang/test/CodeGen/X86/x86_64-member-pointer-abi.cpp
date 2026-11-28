// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

struct S { int i; };
typedef int S::*dmp_t;

// A data member pointer is a scalar, so an atomic one is passed directly as
// its offset.
void take_atomic_dmp(_Atomic(dmp_t) p) {}

// CHECK-LABEL: define dso_local void @_Z15take_atomic_dmpU7_AtomicM1Si(
// CHECK-SAME: i64 %{{.*}})

_Atomic(dmp_t) ret_atomic_dmp() { return nullptr; }

// CHECK-LABEL: define dso_local i64 @_Z14ret_atomic_dmpv()
