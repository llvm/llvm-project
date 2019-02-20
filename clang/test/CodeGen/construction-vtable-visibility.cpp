// RUN: %clang_cc1 -triple x86_64-linux-unknown -fvisibility hidden -emit-llvm %s -o - | FileCheck %s

struct Base {};

class Parent1 : virtual public Base {};

class Parent2 : virtual public Base {};

class Child : public Parent1, public Parent2 {};

void test() {
  Child x;
}

// CHECK: @_ZTC5Child0_7Parent1 = linkonce_odr hidden unnamed_addr constant
// CHECK: @_ZTC5Child8_7Parent2 = linkonce_odr hidden unnamed_addr constant
