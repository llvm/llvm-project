// RUN: echo "fun:*foo*" > %t.always-instrument
// RUN: echo "src:*xray-always-instrument.cpp" >> %t.always-instrument
// RUN: echo "[always]" > %t.xray-attrlist
// RUN: echo "fun:*foo*" >> %t.xray-attrlist
// RUN: echo "src:*xray-always-instrument.cpp" >> %t.xray-attrlist
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 \
// RUN:     -fxray-always-instrument=%t.always-instrument -emit-llvm -o - %s \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 \
// RUN:     -fxray-attr-list=%t.xray-attrlist -emit-llvm -o - %s \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s

void foo() {}

[[clang::xray_never_instrument]] void bar() {}

void baz() {}

// CHECK: define{{.*}} void @_Z3foov() #[[ALWAYSATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z3barv() #[[NEVERATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z3bazv() #[[ALWAYSATTR:[0-9]+]] {
// CHECK: attributes #[[ALWAYSATTR]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// CHECK: attributes #[[NEVERATTR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
