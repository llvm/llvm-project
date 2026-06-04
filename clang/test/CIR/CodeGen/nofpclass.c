// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs -menable-no-nans -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs -menable-no-nans -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs -menable-no-nans -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

float identity(float x) { return x; }
// CIR: cir.func {{.*}} @identity(%arg0: !cir.float {llvm.nofpclass = 519 : i64, llvm.noundef}
// CIR-SAME: -> (!cir.float {llvm.nofpclass = 519 : i64})
// LLVM: define {{.*}} nofpclass(nan inf) float @identity(float noundef nofpclass(nan inf) %{{.*}})

double add(double a, double b) { return a + b; }
// CIR: cir.func {{.*}} @add(%arg0: !cir.double {llvm.nofpclass = 519 : i64, llvm.noundef}
// CIR-SAME: %arg1: !cir.double {llvm.nofpclass = 519 : i64, llvm.noundef}
// CIR-SAME: -> (!cir.double {llvm.nofpclass = 519 : i64})
// LLVM: define {{.*}} nofpclass(nan inf) double @add(double noundef nofpclass(nan inf) %{{.*}}, double noundef nofpclass(nan inf) %{{.*}})

_Complex double ret_complex(void) { return 1.0 + 2.0i; }
// CIR: cir.func {{.*}} @ret_complex() -> (!cir.complex<!cir.double> {llvm.nofpclass = 519 : i64})
// LLVM: define {{.*}} nofpclass(nan inf) { double, double } @ret_complex()

int non_fp(int x) { return x; }
// CIR: cir.func {{.*}} @non_fp(%arg0: !s32i {llvm.noundef}{{.*}}) -> !s32i
// LLVM: define {{.*}} i32 @non_fp(i32 noundef %{{.*}})
