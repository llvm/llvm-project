// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=i386-pc-win32 -fno-rtti -fprofile-instrument=clang | FileCheck %s --check-prefix=GEN
//
// Don't crash when presented profile data for functions without bodies:
// RUN: llvm-profdata merge %S/Inputs/cxx-missing-bodies.proftext -o %t.profdata
// RUN: %clang_cc1 %s -std=c++11 -emit-llvm-only -triple=i386-pc-win32 -fno-rtti -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -w

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
