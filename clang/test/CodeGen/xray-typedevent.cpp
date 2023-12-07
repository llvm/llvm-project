// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @_Z16alwaysInstrumentv
[[clang::xray_always_instrument]] void alwaysInstrument() {
  // Event types would normally come from calling __xray_register_event_type
  // from compiler-rt
  auto EventType = 1;
  static constexpr char kPhase[] = "instrument";
  __xray_typedevent(EventType, kPhase, 10);
  // CHECK: call void @llvm.xray.typedevent(i64 {{.*}}, ptr{{.*}}, i64 10)
}

// CHECK-LABEL: @_Z15neverInstrumentv
[[clang::xray_never_instrument]] void neverInstrument() {
  auto EventType = 2;
  static constexpr char kPhase[] = "never";
  __xray_typedevent(EventType, kPhase, 5);
  // CHECK-NOT: call void @llvm.xray.typedevent(i64 {{.*}}, ptr{{.*}}, i64 5)
}

// CHECK-LABEL: @_Z21conditionalInstrumenti
[[clang::xray_always_instrument]] void conditionalInstrument(int v) {
  auto TrueEventType = 3;
  auto UntrueEventType = 4;
  static constexpr char kTrue[] = "true";
  static constexpr char kUntrue[] = "untrue";
  if (v % 2)
    __xray_typedevent(TrueEventType, kTrue, 4);
  else
    __xray_typedevent(UntrueEventType, kUntrue, 6);

  // CHECK: call void @llvm.xray.typedevent(i64 {{.*}}, ptr{{.*}}, i64 4)
  // CHECK: call void @llvm.xray.typedevent(i64 {{.*}}, ptr{{.*}}, i64 6)
}
