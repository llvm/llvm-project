// RUN: %clang_cc1 -triple  x86_64-unknown-unknown -emit-llvm  \
// RUN: < %s  | FileCheck %s --check-prefixes=CHECK

_Float16 add1(_Float16 a, _Float16 b) {
  // CHECK-LABEL: define{{.*}} half @add1
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: load half, half*
  // CHECK: load half, half* {{.*}}
  // CHECK: fadd half {{.*}}, {{.*}}
  // CHECK: ret half
  return a + b;
}

_Float16 add2(_Float16 a, _Float16 b, _Float16 c) {
  // CHECK-LABEL: define{{.*}} half @add2
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: load half, half* {{.*}}
  // CHECK: load half, half* {{.*}}
  // CHECK: fadd half {{.*}}, {{.*}}
  // CHECK: load half, half* {{.*}}
  // CHECK: fadd half {{.*}}, {{.*}}
  // CHECK: ret half
    return a + b + c;
}

_Float16 sub(_Float16 a, _Float16 b) {
  // CHECK-LABEL: define{{.*}} half @sub
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: load half, half*
  // CHECK: load half, half* {{.*}}
  // CHECK: fsub half {{.*}}, {{.*}}
  // CHECK: ret half
  return a - b;
}

_Float16 div(_Float16 a, _Float16 b) {
  // CHECK-LABEL: define{{.*}} half @div
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: load half, half* {{.*}}
  // CHECK: load half, half* {{.*}}
  // CHECK: fdiv half {{.*}}, {{.*}}
  // CHECK: ret half
  return a / b;
}

_Float16 mul(_Float16 a, _Float16 b) {
  // CHECK-LABEL: define{{.*}} half @mul
  // CHECK: alloca half
  // CHECK: alloca half
  // CHECK: store half {{.*}}, half*
  // CHECK: store half {{.*}}, half*
  // CHECK: load half, half* {{.*}}
  // CHECK: load half, half* {{.*}}
  // CHECK: fmul half {{.*}}, {{.*}}
  // CHECK: ret half
  return a * b;
}


