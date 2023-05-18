// RUN: %clang_cc1 -emit-llvm %s -std=c++11 -S -emit-llvm -o - -triple=i386-pc-win32 -fno-rtti -fprofile-instrument=clang | FileCheck %s --check-prefix=GEN
//
// Don't crash when presented profile data for functions without bodies:
// RUN: llvm-profdata merge %S/Inputs/cxx-missing-bodies.proftext -o %t.profdata
// RUN: %clang_cc1 -emit-llvm %s -std=c++11 -S -emit-llvm -o /dev/null -triple=i386-pc-win32 -fno-rtti -fprofile-instrument-use-path=%t.profdata -w

// GEN-NOT: __profn{{.*}}??_GA@@UAEPAXI@Z
// GEN-NOT: __profn{{.*}}??_DA@@QAEXXZ

struct A {
  virtual ~A();
};
struct B : A {
  virtual ~B();
};

B::~B() {}

void foo() {
  B c;
}
