// RUN: %clang_cc1 %s -std=c++14 -triple=spir -verify -fsyntax-only
// RUN: %clang_cc1 %s -std=c++17 -triple=spir -verify -fsyntax-only

struct MyType {
  MyType(int i) : i(i) {}
  int i;
};

//expected-note@-5{{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const MyType &' for 1st argument}}
//expected-note@-6{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'MyType &&' for 1st argument}}
//expected-note@-6{{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(10)))'}}
//expected-note@-8{{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(10)))'}}
//expected-note@-9{{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(10)))'}}
//expected-note@-9{{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(10)))'}}

// FIXME: We can't implicitly convert between address spaces yet.
MyType __attribute__((address_space(10))) m1 = 123; //expected-error{{no viable conversion from 'int' to '__attribute__((address_space(10))) MyType'}}
MyType __attribute__((address_space(10))) m2(123);  //expected-error{{no matching constructor for initialization of '__attribute__((address_space(10))) MyType'}}
