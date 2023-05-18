// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s 

struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) IUnknown {
  void foo();
};

__interface NoError : public IUnknown {};
void IUnknown::foo() {}
// expected-error@+1{{interface type cannot inherit from}}
__interface HasError : public IUnknown {}; 
