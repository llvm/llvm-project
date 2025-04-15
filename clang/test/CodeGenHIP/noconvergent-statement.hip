// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -fconvergence-control -emit-llvm -disable-llvm-optzns -fcuda-is-device -o - %s | FileCheck %s

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

__device__ bool check() __attribute__((convergent));

// ------ The entry of a noconvergent function uses an anchor
// CHECK-LABEL: @_Z3bari
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK0]]) ]
__device__ __attribute__((noconvergent)) int bar(int x) {
  foo();
  return x;
}

// ------ No token on a call to a noconvergent function
// CHECK-LABEL: @_Z17call_noconvergenti
// CHECK:  %call = call noundef i32 @_Z3bari(i32 noundef %4) #
__device__ int call_noconvergent(int cond) {
  int i = 0;

  while (i < cond) {
    i = bar(i);
  }

  return i;
}

// CHECK-LABEL: @_Z16branch_statementi
// CHECK: noconvergent.anchor:
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   %call1 = call noundef zeroext i1 @_Z5checkv() #4 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: if.then:
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK1]]) ]
__device__ int branch_statement(int cond) {
  int i = 0;

  [[clang::noconvergent]] if (check()) {
    foo();
  } else {
    foo();
  }

  return i;
}

// CHECK-LABEL: @_Z19branch_substatementi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:   %call1 = call noundef zeroext i1 @_Z5checkv() #4 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: noconvergent.anchor:
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: noconvergent.anchor2:
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK2]]) ]
__device__ int branch_substatement(int cond) {
  int i = 0;

  if (check()) [[clang::noconvergent]] {
    foo();
  } else [[clang::noconvergent]] {
    foo();
  }

  return i;
}

// CHECK-LABEL: @_Z9loop_bodyi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:                                       ; preds = %noconvergent.anchor, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK:   %call1 = call noundef zeroext i1 @_Z5checkv() #4 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: noconvergent.anchor:                              ; preds = %while.body
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK2]]) ]
__device__ int loop_body(int cond) {
  int i = 0;

  while (check()) [[clang::noconvergent]] {
    foo();
    i++;
  }

  return i;
}

// CHECK-LABEL: @_Z14loop_statementi
// CHECK: noconvergent.anchor:
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK: while.cond:
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK1]]) ]
// CHECK:   %call1 = call noundef zeroext i1 @_Z5checkv() #4 [ "convergencectrl"(token [[TOK2]]) ]
// CHECK: while.body:
// CHECK:   call void @_Z3foov() #4 [ "convergencectrl"(token [[TOK2]]) ]
__device__ int loop_statement(int cond) {
  int i = 0;

  [[clang::noconvergent]] while (check()) {
    foo();
    i++;
  }

  return i;
}
