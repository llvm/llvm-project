// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

bool cond();
void foo();

void test1() {
  do {
  } while (cond());
}
// CHECK: define spir_func void @_Z5test1v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: do.body:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK: do.cond:
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test2() {
  do {
    foo();
  } while (cond());
}
// CHECK: define spir_func void @_Z5test2v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: do.body:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: do.cond:
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test3() {
  do {
    if (cond())
      foo();
  } while (cond());
}
// CHECK: define spir_func void @_Z5test3v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: do.body:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK: if.then:
// CHECK:                    call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: do.cond:
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test4() {
  do {
    if (cond()) {
      foo();
      break;
    }
  } while (cond());
}
// CHECK: define spir_func void @_Z5test4v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: do.body:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK: if.then:
// CHECK:                    call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: do.cond:
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test5() {
  do {
    while (cond()) {
      if (cond()) {
        foo();
        break;
      }
    }
  } while (cond());
}
// CHECK: define spir_func void @_Z5test5v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: do.body:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK: while.cond:
// CHECK:   [[T2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T1]]) ]
// CHECK: if.then:
// CHECK:                    call spir_func void @_Z3foov() [[A3]] [ "convergencectrl"(token [[T2]]) ]
// CHECK: do.cond:
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

// CHECK-DAG: attributes [[A0]] = { {{.*}}convergent{{.*}} }
// CHECK-DAG: attributes [[A3]] = { {{.*}}convergent{{.*}} }
