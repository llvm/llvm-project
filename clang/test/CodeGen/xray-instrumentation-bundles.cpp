// RUN: %clang_cc1 -fxray-instrument -fxray-instrumentation-bundle=none -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,NOFUNCTION,NOCUSTOM %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,FUNCTION,NOCUSTOM %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=custom -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,NOFUNCTION,CUSTOM %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function,custom -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,FUNCTION,CUSTOM %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function \
// RUN:     -fxray-instrumentation-bundle=custom -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,FUNCTION,CUSTOM %s

// CHECK: define void @_Z16alwaysInstrumentv() #[[ALWAYSATTR:[0-9]+]] {
[[clang::xray_always_instrument]] void alwaysInstrument() {
  static constexpr char kPhase[] = "always";
  __xray_customevent(kPhase, 6);
  // CUSTOM: call void @llvm.xray.customevent(i8*{{.*}}, i32 6)
  // NOCUSTOM-NOT: call void @llvm.xray.customevent(i8*{{.*}}, i32 6)
}

// FUNCTION: attributes #[[ALWAYSATTR]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// NOFUNCTION-NOT: attributes #[[ALWAYSATTR]] = {{.*}} "function-instrument"="xray-always" {{.*}}
