// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST1
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify -DTEST2

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

// We expect no warnings here, as there is only declaration of _ZN1TD1Ev function, no definitions.
extern "C" void _ZN1TD1Ev();
struct T {
  ~T() {}
};

void foo() {
  _ZN1TD1Ev();
  T t;
}

extern "C" void _ZN2T2D2Ev() {}; // expected-note {{previous definition is here}}

struct T2 {
  ~T2() {} // expected-error {{definition with same mangled name as another definition}}
};

void bar() {
  _ZN2T2D2Ev();
  T2 t;
}

#else

#error Unknwon test

#endif

