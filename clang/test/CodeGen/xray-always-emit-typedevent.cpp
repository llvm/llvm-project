// RUN: %clang_cc1 -fxray-instrument -fxray-always-emit-typedevents -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck %s

// CHECK-LABEL: @_Z15neverInstrumentv
[[clang::xray_never_instrument]] void neverInstrument() {
  static constexpr char kPhase[] = "never";
  __xray_typedevent(1, kPhase, 5);
  // CHECK: call void @llvm.xray.typedevent(i64 {{.*}}, ptr{{.*}}, i64 5)
}
