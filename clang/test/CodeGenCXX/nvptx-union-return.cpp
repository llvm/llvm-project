// RUN: %clang_cc1 -triple nvptx64 -emit-llvm -o - %s | FileCheck %s

// A union in a base class is returned as an integer array, like a union
// member, so that bytes of the active member are not dropped.
union U {
  unsigned long long l;
  struct {
    short s;
    unsigned long long l;
  } p;
};

struct Base {
  U u;
};

struct Derived : Base {
  int i;
};

// The original packed LLVM return type has byte alignment.
// CHECK-LABEL: define{{.*}} [24 x i8] @_Z11ret_derivedv
Derived ret_derived() { return Derived(); }
