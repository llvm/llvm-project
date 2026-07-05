// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck %s

struct A {
  A();
  ~A();
};

A a;
A as[2];

struct B {
  B();
  ~B();
  int f();
};

int i = B().f();

// CHECK: "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: "__TEXT,__StaticInit,regular,pure_instructions" {
