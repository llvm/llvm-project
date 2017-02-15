// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -main-file-name cxx-virtual-destructor-calls.cpp %s -o - -fprofile-instrument=clang | FileCheck %s

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};

// Base dtor counters and profile data
// CHECK: @__profc__ZN1BD2Ev = private global [1 x i64] zeroinitializer
// CHECK: @__profd__ZN1BD2Ev =

// Complete dtor counters and profile data must absent
// CHECK-NOT: @__profc__ZN1BD1Ev = private global [1 x i64] zeroinitializer
// CHECK-NOT: @__profd__ZN1BD1Ev =

// Deleting dtor counters and profile data must absent
// CHECK-NOT: @__profc__ZN1BD0Ev = private global [1 x i64] zeroinitializer
// CHECK-NOT: @__profd__ZN1BD0Ev =

B::~B() { }
