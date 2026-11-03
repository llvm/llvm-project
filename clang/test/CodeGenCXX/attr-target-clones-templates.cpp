// RUN: %clang_cc1 -std=c++14 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct InlineDependency {
  InlineDependency(int &value) : value(value) {}
  int &value;
};

template <typename T>
T __attribute__((target_clones("sse4.2", "default"))) templated(T value) {
  return value;
}

int call_int(int value) {
  return templated(value);
}

float call_float(float value) {
  return templated(value);
}

int (*address_int())(int) {
  return &templated<int>;
}

template double templated<double>(double);

template <typename T>
T __attribute__((target_clones("sse4.2", "default")))
with_inline_dependency(T &value) {
  InlineDependency dependency(value);
  return dependency.value;
}

int call_with_inline_dependency(int &value) {
  return with_inline_dependency(value);
}

// CHECK-DAG: $_Z9templatedIiET_S0_.resolver = comdat any

// CHECK-DAG: $_Z9templatedIfET_S0_.resolver = comdat any
// CHECK-DAG: $_Z9templatedIdET_S0_.resolver = comdat any
// CHECK-DAG: $_Z9templatedIiET_S0_.sse4.2.0 = comdat any
// CHECK-DAG: $_Z9templatedIiET_S0_.default.1 = comdat any
// CHECK-DAG: $_Z9templatedIfET_S0_.sse4.2.0 = comdat any
// CHECK-DAG: $_Z9templatedIfET_S0_.default.1 = comdat any
// CHECK-DAG: $_Z9templatedIdET_S0_.sse4.2.0 = comdat any
// CHECK-DAG: $_Z9templatedIdET_S0_.default.1 = comdat any
// CHECK-DAG: $_Z22with_inline_dependencyIiET_RS0_.resolver = comdat any
// CHECK-DAG: $_Z22with_inline_dependencyIiET_RS0_.sse4.2.0 = comdat any
// CHECK-DAG: $_Z22with_inline_dependencyIiET_RS0_.default.1 = comdat any

// CHECK-DAG: @_Z9templatedIiET_S0_.ifunc = weak_odr alias i32 (i32), ptr @_Z9templatedIiET_S0_
// CHECK-DAG: @_Z9templatedIfET_S0_.ifunc = weak_odr alias float (float), ptr @_Z9templatedIfET_S0_
// CHECK-DAG: @_Z9templatedIdET_S0_.ifunc = weak_odr alias double (double), ptr @_Z9templatedIdET_S0_
// CHECK-DAG: @_Z9templatedIiET_S0_ = weak_odr ifunc i32 (i32), ptr @_Z9templatedIiET_S0_.resolver
// CHECK-DAG: @_Z9templatedIfET_S0_ = weak_odr ifunc float (float), ptr @_Z9templatedIfET_S0_.resolver
// CHECK-DAG: @_Z9templatedIdET_S0_ = weak_odr ifunc double (double), ptr @_Z9templatedIdET_S0_.resolver

// CHECK-DAG: @_Z22with_inline_dependencyIiET_RS0_.ifunc = weak_odr alias i32 (ptr), ptr @_Z22with_inline_dependencyIiET_RS0_
// CHECK-DAG: @_Z22with_inline_dependencyIiET_RS0_ = weak_odr ifunc i32 (ptr), ptr @_Z22with_inline_dependencyIiET_RS0_.resolver
// CHECK-LABEL: define dso_local noundef i32 @_Z8call_inti(
// CHECK: call noundef i32 @_Z9templatedIiET_S0_(i32 noundef

// CHECK-LABEL: define weak_odr ptr @_Z9templatedIiET_S0_.resolver() {{.*}} comdat
// CHECK: ret ptr @_Z9templatedIiET_S0_.sse4.2.0
// CHECK: ret ptr @_Z9templatedIiET_S0_.default.1

// CHECK-LABEL: define dso_local noundef float @_Z10call_floatf(
// CHECK: call noundef float @_Z9templatedIfET_S0_(float noundef

// CHECK-LABEL: define weak_odr ptr @_Z9templatedIfET_S0_.resolver() {{.*}} comdat
// CHECK: ret ptr @_Z9templatedIfET_S0_.sse4.2.0
// CHECK: ret ptr @_Z9templatedIfET_S0_.default.1

// CHECK-LABEL: define dso_local noundef ptr @_Z11address_intv(
// CHECK: ret ptr @_Z9templatedIiET_S0_

// CHECK-LABEL: define weak_odr noundef double @_Z9templatedIdET_S0_.sse4.2.0(
// CHECK-LABEL: define weak_odr noundef double @_Z9templatedIdET_S0_.default.1(
// CHECK-LABEL: define weak_odr ptr @_Z9templatedIdET_S0_.resolver() {{.*}} comdat
// CHECK: ret ptr @_Z9templatedIdET_S0_.sse4.2.0
// CHECK: ret ptr @_Z9templatedIdET_S0_.default.1

// CHECK-LABEL: define dso_local noundef i32 @_Z27call_with_inline_dependencyRi(
// CHECK: call noundef i32 @_Z22with_inline_dependencyIiET_RS0_(

// CHECK-LABEL: define weak_odr ptr @_Z22with_inline_dependencyIiET_RS0_.resolver() {{.*}} comdat
// CHECK: ret ptr @_Z22with_inline_dependencyIiET_RS0_.sse4.2.0
// CHECK: ret ptr @_Z22with_inline_dependencyIiET_RS0_.default.1

// CHECK-LABEL: define linkonce_odr noundef i32 @_Z9templatedIiET_S0_.sse4.2.0(
// CHECK-LABEL: define linkonce_odr noundef i32 @_Z9templatedIiET_S0_.default.1(
// CHECK-LABEL: define linkonce_odr noundef float @_Z9templatedIfET_S0_.sse4.2.0(
// CHECK-LABEL: define linkonce_odr noundef float @_Z9templatedIfET_S0_.default.1(
// CHECK-LABEL: define linkonce_odr noundef i32 @_Z22with_inline_dependencyIiET_RS0_.sse4.2.0(
// CHECK-LABEL: define linkonce_odr void @_ZN16InlineDependencyC1ERi(
// CHECK-LABEL: define linkonce_odr noundef i32 @_Z22with_inline_dependencyIiET_RS0_.default.1(
// CHECK-LABEL: define linkonce_odr void @_ZN16InlineDependencyC2ERi(
