// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs -menable-no-nans -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs -menable-no-nans -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

float identity(float x) { return x; }
// LLVM: define {{.*}} nofpclass(nan inf) float @identity(float noundef nofpclass(nan inf) %{{.*}})
// OGCG: define {{.*}} nofpclass(nan inf) float @identity(float noundef nofpclass(nan inf) %{{.*}})

double add(double a, double b) { return a + b; }
// LLVM: define {{.*}} nofpclass(nan inf) double @add(double noundef nofpclass(nan inf) %{{.*}}, double noundef nofpclass(nan inf) %{{.*}})
// OGCG: define {{.*}} nofpclass(nan inf) double @add(double noundef nofpclass(nan inf) %{{.*}}, double noundef nofpclass(nan inf) %{{.*}})

_Complex double ret_complex(void) { return 1.0 + 2.0i; }
// LLVM: define {{.*}} nofpclass(nan inf) { double, double } @ret_complex()
// OGCG: define {{.*}} nofpclass(nan inf) { double, double } @ret_complex()

int non_fp(int x) { return x; }
// LLVM: define {{.*}} i32 @non_fp(i32 noundef %{{.*}})
// LLVM-NOT: nofpclass
// OGCG: define {{.*}} i32 @non_fp(i32 noundef %{{.*}})
// OGCG-NOT: nofpclass
