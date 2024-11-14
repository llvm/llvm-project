// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

bool cond();
void foo();

void test1() {
  while (cond()) {
  }
}
// CHECK: define spir_func void @_Z5test1v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test2() {
  while (cond()) {
    foo();
  }
}
// CHECK: define spir_func void @_Z5test2v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: while.body:
// CHECK:   call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]

void test3() {
  while (cond()) {
    if (cond())
      break;
    foo();
  }
}
// CHECK: define spir_func void @_Z5test3v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: if.then:
// CHECK:   br label %while.end
// CHECK: if.end:
// CHECK:   call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK:   br label %while.cond

void test4() {
  while (cond()) {
    if (cond()) {
      foo();
      break;
    }
  }
}
// CHECK: define spir_func void @_Z5test4v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: if.then:
// CHECK:   call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK:   br label %while.end
// CHECK: if.end:
// CHECK:   br label %while.cond

void test5() {
  while (cond()) {
    while (cond()) {
      if (cond()) {
        foo();
        break;
      }
    }
  }
}
// CHECK: define spir_func void @_Z5test5v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: while.cond2:
// CHECK:   [[T2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T1]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T2]]) ]
// CHECK: if.then:
// CHECK:   call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T2]]) ]
// CHECK:   br label %while.end

void test6() {
  while (cond()) {
    while (cond()) {
    }

    if (cond()) {
      foo();
      break;
    }
  }
}
// CHECK: define spir_func void @_Z5test6v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: while.cond2:
// CHECK:   [[T2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T1]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T2]]) ]
// CHECK: if.then:
// CHECK:   call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK:   br label %while.end

// CHECK-DAG: attributes [[A0]] = { {{.*}}convergent{{.*}} }
// CHECK-DAG: attributes [[A3]] = { {{.*}}convergent{{.*}} }
