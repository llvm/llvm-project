// RUN: echo "fun:*arg1*=arg1" >> %t.always-instrument
// RUN: %clang_cc1 -fxray-instrument -x c++ -std=c++11 -fxray-always-instrument=%t.always-instrument -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck %s

void foo() {}

void arg1(void*) {}

// CHECK: define void @_Z3foov() #[[FOO:[0-9]+]] {
// CHECK: define void {{.*}}arg1{{.*}} #[[ALWAYSARG1:[0-9]+]] {

// CHECK: attributes #[[FOO]] = {{.*}}
// CHECK: attributes #[[ALWAYSARG1]] = {{.*}} "function-instrument"="xray-always" {{.*}} "xray-log-args"="1"
