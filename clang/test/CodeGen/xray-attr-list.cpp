// RUN: echo "[always]" > %t.xray-attrlist
// RUN: echo "fun:*always*" >> %t.xray-attrlist
// RUN: echo "[never]" >> %t.xray-attrlist
// RUN: echo "fun:*never*" >> %t.xray-attrlist
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 \
// RUN:     -fxray-attr-list=%t.xray-attrlist -emit-llvm -o - %s \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s

void always() {}
void never() {}
[[clang::xray_never_instrument]] void alwaysNever() {}
[[clang::xray_always_instrument]] void neverAlways() {}

// CHECK: define{{.*}} void @_Z6alwaysv() #[[ALWAYSATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z5neverv() #[[NEVERATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z11alwaysNeverv() #[[NEVERATTR]] {
// CHECK: define{{.*}} void @_Z11neverAlwaysv() #[[ALWAYSATTR]] {
// CHECK: attributes #[[ALWAYSATTR]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// CHECK: attributes #[[NEVERATTR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
