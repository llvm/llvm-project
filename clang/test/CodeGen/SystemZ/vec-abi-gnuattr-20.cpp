// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Globally visible C++ object with vector member.

typedef __attribute__((vector_size(16))) int v4i32;

class Base {
protected:
  v4i32 v;
};

class C : public Base {
  int i;

public:
  C() {
    i = 1;
    v = {1, 2, 3, 4};
  }
};

C Obj;

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
