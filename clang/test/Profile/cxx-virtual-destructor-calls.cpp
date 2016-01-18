// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -main-file-name cxx-virtual-destructor-calls.cpp %s -o - -fprofile-instr-generate | FileCheck %s

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

// Complete dtor
// CHECK: @__llvm_profile_name__ZN1BD1Ev = private constant [9 x i8] c"_ZN1BD1Ev"

// Deleting dtor
// CHECK: @__llvm_profile_name__ZN1BD0Ev = private constant [9 x i8] c"_ZN1BD0Ev"

// Complete dtor counters and profile data
// CHECK: @__llvm_profile_counters__ZN1BD1Ev = private global [1 x i64] zeroinitializer
// CHECK: @__llvm_profile_data__ZN1BD1Ev =

// Deleting dtor counters and profile data
// CHECK: @__llvm_profile_counters__ZN1BD0Ev = private global [1 x i64] zeroinitializer
// CHECK: @__llvm_profile_data__ZN1BD0Ev =

B::~B() { }
