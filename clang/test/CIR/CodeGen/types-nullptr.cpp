// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef decltype(nullptr) nullptr_t;
void f() { nullptr_t t = nullptr; }

// CHECK: %0 = cir.alloca !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>
// CHECK: %1 = cir.const(#cir.ptr<null> : !cir.ptr<!void>) : !cir.ptr<!void>
// CHECK: cir.store %1, %0 : !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>
