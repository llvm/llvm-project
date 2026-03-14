// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

void cb0(void);

// CHECK-DAG: !callback ![[cid0:[0-9]+]] void @no_args
__attribute__((callback(1))) void no_args(void (*callback)(void));

// CHECK-DAG: @args_1({{[^#]*#[0-9]+}} !callback ![[cid1:[0-9]+]]
__attribute__((callback(1, 2, 3))) void args_1(void (*callback)(int, double), int a, double b) { no_args(cb0); }

// CHECK-DAG: !callback ![[cid2:[0-9]+]]  void @args_2a
__attribute__((callback(2, 3, 3))) void args_2a(int a, void (*callback)(double, double), double b);
// CHECK-DAG: !callback ![[cid2]]         void @args_2b
__attribute__((callback(callback, b, b))) void args_2b(int a, void (*callback)(double, double), double b);

// CHECK-DAG: void @args_3a({{[^#]*#[0-9]+}} !callback ![[cid3:[0-9]+]]
__attribute__((callback(2, -1, -1))) void args_3a(int a, void (*callback)(double, double), double b) { args_2a(a, callback, b); }
// CHECK-DAG: void @args_3b({{[^#]*#[0-9]+}} !callback ![[cid3]]
__attribute__((callback(callback, __, __))) void args_3b(int a, void (*callback)(double, double), double b) { args_2b(a, callback, b); }

// CHECK-DAG: ![[cid0]] = !{![[cid0b:[0-9]+]]}
// CHECK-DAG: ![[cid0b]] = !{i64 0, i1 false}
// CHECK-DAG: ![[cid1]] = !{![[cid1b:[0-9]+]]}
// CHECK-DAG: ![[cid1b]] = !{i64 0, i64 1, i64 2, i1 false}
// CHECK-DAG: ![[cid2]] = !{![[cid2b:[0-9]+]]}
// CHECK-DAG: ![[cid2b]] = !{i64 1, i64 2, i64 2, i1 false}
// CHECK-DAG: ![[cid3]] = !{![[cid3b:[0-9]+]]}
// CHECK-DAG: ![[cid3b]] = !{i64 1, i64 -1, i64 -1, i1 false}
