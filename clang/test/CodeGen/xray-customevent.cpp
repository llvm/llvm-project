// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @_Z16alwaysInstrumentv
[[clang::xray_always_instrument]] void alwaysInstrument() {
  static constexpr char kPhase[] = "instrument";
  __xray_customevent(kPhase, 10);
  // CHECK: call void @llvm.xray.customevent(ptr{{.*}}, i32 10)
}

// CHECK-LABEL: @_Z15neverInstrumentv
[[clang::xray_never_instrument]] void neverInstrument() {
  static constexpr char kPhase[] = "never";
  __xray_customevent(kPhase, 5);
  // CHECK-NOT: call void @llvm.xray.customevent(ptr{{.*}}, i32 5)
}

// CHECK-LABEL: @_Z21conditionalInstrumenti
[[clang::xray_always_instrument]] void conditionalInstrument(int v) {
  static constexpr char kTrue[] = "true";
  static constexpr char kUntrue[] = "untrue";
  if (v % 2)
    __xray_customevent(kTrue, 4);
  else
    __xray_customevent(kUntrue, 6);

  // CHECK: call void @llvm.xray.customevent(ptr{{.*}}, i32 4)
  // CHECK: call void @llvm.xray.customevent(ptr{{.*}}, i32 6)
}
