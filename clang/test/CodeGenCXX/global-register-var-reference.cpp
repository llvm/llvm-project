// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

register unsigned long sp asm("rsp");

template <class T> unsigned long f(const T &x) { return x; }

// A const reference to a global register variable binds to a temporary that
// holds the register's value.
// CHECK-LABEL: define {{.*}} @_Z1gv(
// CHECK:         [[TMP:%.*]] = alloca i64
// CHECK:         [[VAL:%.*]] = call i64 @llvm.read_register.i64(metadata ![[REG:[0-9]+]])
// CHECK-NEXT:    store i64 [[VAL]], ptr [[TMP]]
// CHECK:         call {{.*}} @_Z1fImEmRKT_(ptr {{.*}} [[TMP]])
unsigned long g() { return f(sp); }

// CHECK: ![[REG]] = !{!"rsp"}
