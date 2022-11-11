// RUN: %clang_cc1 -O2 -emit-llvm -fms-extensions -o - %s | FileCheck %s

#pragma optimize("", off)

// CHECK: define{{.*}} void @f0(){{.*}} #[[OPTNONE:[0-9]+]]
void f0() {}

// CHECK: define{{.*}} void @f1(){{.*}} #[[OPTNONE]]
void f1() {}

#pragma optimize("", on)

// CHECK: define{{.*}} void @f2(){{.*}} #[[NO_OPTNONE:[0-9]+]]
void f2() {}

// CHECK: define{{.*}} void @f3(){{.*}} #[[NO_OPTNONE]]
void f3() {}

// CHECK:         attributes #[[OPTNONE]] = {{{.*}}optnone{{.*}}}
// CHECK-NOT:     attributes #[[NO_OPTNONE]] = {{{.*}}optnone{{.*}}}
