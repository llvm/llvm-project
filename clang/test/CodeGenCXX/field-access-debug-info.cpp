// RUN: %clang_cc1 -g -S -masm-verbose -o - %s | FileCheck %s

// CHECK: abbrev_begin:
// CHECK: DW_AT_accessibility
// CHECK-NEXT: DW_FORM_data1

class A {
public:
  int p;
private:
  int pr;
};

A a;
