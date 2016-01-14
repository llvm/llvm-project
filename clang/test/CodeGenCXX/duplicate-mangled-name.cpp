// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST1
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST3
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST4

#ifdef TEST1

// rdar://15522601
class MyClass {
 static void meth();
};
void MyClass::meth() { } // expected-note {{previous}}
extern "C" {
  void _ZN7MyClass4methEv() { } // expected-error {{definition with same mangled name as another definition}}
}

#elif TEST2

// expected-no-diagnostics

// We expect no warnings here, as there is only declaration of _ZN1TD1Ev
// function, no definitions.
extern "C" void _ZN1TD1Ev();
struct T {
  ~T() {}
};

// We expect no warnings here, as there is only declaration of _ZN2nm3abcE
// global, no definitions.
extern "C" {
  int _ZN2nm3abcE;
}

namespace nm {
  float abc = 2;
}
// CHECK: @_ZN2nm3abcE = global float

float foo() {
  _ZN1TD1Ev();
// CHECK: call void bitcast ({{.*}} (%struct.T*)* @_ZN1TD1Ev to void ()*)()
  T t;
// CHECK: call {{.*}} @_ZN1TD1Ev(%struct.T* %t)
  return _ZN2nm3abcE + nm::abc;
}

#elif TEST3

extern "C" void _ZN2T2D2Ev() {}; // expected-note {{previous definition is here}}

struct T2 {
  ~T2() {} // expected-error {{definition with same mangled name as another definition}}
};

void foo() {
  _ZN2T2D2Ev();
  T2 t;
}

#elif TEST4

extern "C" {
  int _ZN2nm3abcE = 1; // expected-note {{previous definition is here}}
}

namespace nm {
  float abc = 2; // expected-error {{definition with same mangled name as another definition}}
}

float foo() {
  return _ZN2nm3abcE + nm::abc;
}

#else

#error Unknwon test

#endif

