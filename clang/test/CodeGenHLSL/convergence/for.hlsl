// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

bool cond();
bool cond2();
void foo();

void test1() {
  for (;;) {
    foo();
  }
}
// CHECK: define spir_func void @_Z5test1v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test2() {
  for (;cond();) {
    foo();
  }
}
// CHECK: define spir_func void @_Z5test2v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: for.body:
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test3() {
  for (cond();;) {
    foo();
  }
}
// CHECK: define spir_func void @_Z5test3v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T0]]) ]
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test4() {
  for (cond();cond2();) {
    foo();
  }
}
// CHECK: define spir_func void @_Z5test4v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T0]]) ]
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z5cond2v() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: for.body:
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test5() {
  for (cond();cond2();foo()) {
  }
}
// CHECK: define spir_func void @_Z5test5v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T0]]) ]
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z5cond2v() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: for.inc:
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test6() {
  for (cond();cond2();foo()) {
    if (cond()) {
      foo();
      break;
    }
  }
}
// CHECK: define spir_func void @_Z5test6v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T0]]) ]
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z5cond2v() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: for.body:
// CHECK:   [[C1:%[a-zA-Z0-9]+]] = call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK:   br i1 [[C1]], label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]
// CHECK:   br label %for.end
// CHECK: if.end:
// CHECK:   br label %for.inc
// CHECK: for.inc:
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T1]]) ]

void test7() {
  for (cond();;) {
    for (cond();;) {
      foo();
    }
  }
}
// CHECK: define spir_func void @_Z5test7v() [[A0:#[0-9]+]] {
// CHECK: entry:
// CHECK:   [[T0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T0]]) ]
// CHECK: for.cond:
// CHECK:   [[T1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T0]]) ]
// CHECK:                    call spir_func noundef i1 @_Z4condv() [[A3]] [ "convergencectrl"(token [[T1]]) ]
// CHECK: for.cond3:
// CHECK:   [[T2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[T1]]) ]
// CHECK:                    call spir_func void @_Z3foov() [[A3:#[0-9]+]] [ "convergencectrl"(token [[T2]]) ]

// CHECK-DAG: attributes [[A0]] = { {{.*}}convergent{{.*}} }
// CHECK-DAG: attributes [[A3]] = { {{.*}}convergent{{.*}} }
