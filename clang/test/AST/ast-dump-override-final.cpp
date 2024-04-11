// This file contain tests to check if override and final are dumped in the
// correct positions.

// RUN: %clang_cc1 -ast-print -x c++ %s -o - | FileCheck %s

// CHECK: class A {
class A {
  // CHECK-NEXT: virtual void f();
  virtual void f();

  // CHECK-NEXT: virtual void g() final;
  virtual void g() final;
} AA;

// CHECK: class B : public A {
class B : public A {
  // CHECK-NEXT: virtual void f() override {
  virtual void f() override {
  };
} B;
