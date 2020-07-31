// RUN: %clang_cc1 -emit-llvm -triple powerpc-ibm-aix-xcoff -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -emit-llvm -triple powerpc64-ibm-aix-xcoff -x c++ < %s | \
// RUN:   FileCheck %s

struct A {
  char x;
};

struct B {
  int x;
};

struct __attribute__((__packed__)) C : A, B {} c;

int s = sizeof(c);

// CHECK: @c = global %struct.C zeroinitializer, align 1
// CHECK: @s = global i32 5
