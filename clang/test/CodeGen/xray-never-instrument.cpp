// RUN: echo "fun:*foo*" > %t.never-instrument
// RUN: echo "src:*xray-never-instrument.cpp" >> %t.never-instrument
// RUN: echo "[never]" > %t.xray-attrlist
// RUN: echo "fun:*foo*" >> %t.xray-attrlist
// RUN: echo "src:*xray-never-instrument.cpp" >> %t.xray-attrlist
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 \
// RUN:     -fxray-never-instrument=%t.never-instrument -emit-llvm -o - %s \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 \
// RUN:     -fxray-attr-list=%t.xray-attrlist -emit-llvm -o - %s \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s

void foo() {}

[[clang::xray_always_instrument]] void bar() {}

void baz() {}

// CHECK: define{{.*}} void @_Z3foov() #[[NEVERATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z3barv() #[[ALWAYSATTR:[0-9]+]] {
// CHECK: define{{.*}} void @_Z3bazv() #[[NEVERATTR:[0-9]+]] {
// CHECK: attributes #[[NEVERATTR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
// CHECK: attributes #[[ALWAYSATTR]] = {{.*}} "function-instrument"="xray-always" {{.*}}

