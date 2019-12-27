//RUN: %clang_cc1 %s -cl-std=clc++ -pedantic -ast-dump -verify | FileCheck %s

//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'int (__private int) const __generic'
auto glambda = [](auto a) { return a; };

__kernel void test() {
  int i;
//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'void () const __generic'
  auto  llambda = [&]() {i++;};
  llambda();
  glambda(1);
  // Test lambda with default parameters
//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'void () const __generic'
  [&] {i++;} ();
  __constant auto err = [&]() {}; //expected-note{{candidate function not viable: 'this' object is in address space '__constant', but method expects object in address space '__generic'}}
  err();                          //expected-error-re{{no matching function for call to object of type '__constant (lambda at {{.*}})'}}
  // FIXME: There is very limited addr space functionality
  // we can test when taking lambda type from the object.
  // The limitation is due to addr spaces being added to all
  // objects in OpenCL. Once we add metaprogramming utility
  // for removing address spaces from a type we can enhance
  // testing here.
  (*(__constant decltype(llambda) *)nullptr)(); //expected-error{{multiple address spaces specified for type}}
  (*(decltype(llambda) *)nullptr)();
}

__kernel void test_qual() {
//CHECK: |-CXXMethodDecl {{.*}} constexpr operator() 'void () const __private'
  auto priv1 = []() __private {};
  priv1();
//CHECK: |-CXXMethodDecl {{.*}} constexpr operator() 'void () const __generic'
  auto priv2 = []() __generic {};
  priv2();
  auto priv3 = []() __global {}; //expected-note{{candidate function not viable: 'this' object is in address space '__private', but method expects object in address space '__global'}} //expected-note{{conversion candidate of type 'void (*)()'}}
  priv3(); //expected-error{{no matching function for call to object of type}}

  __constant auto const1 = []() __private{}; //expected-note{{candidate function not viable: 'this' object is in address space '__constant', but method expects object in address space '__private'}} //expected-note{{conversion candidate of type 'void (*)()'}}
  const1(); //expected-error{{no matching function for call to object of type '__constant (lambda at}}
  __constant auto const2 = []() __generic{}; //expected-note{{candidate function not viable: 'this' object is in address space '__constant', but method expects object in address space '__generic'}} //expected-note{{conversion candidate of type 'void (*)()'}}
  const2(); //expected-error{{no matching function for call to object of type '__constant (lambda at}}
//CHECK: |-CXXMethodDecl {{.*}} constexpr operator() 'void () const __constant'
  __constant auto const3 = []() __constant{};
  const3();

  [&] () __global {} (); //expected-error{{no matching function for call to object of type '(lambda at}} expected-note{{candidate function not viable: 'this' object is in default address space, but method expects object in address space '__global'}}
  [&] () __private {} (); //expected-error{{no matching function for call to object of type '(lambda at}} expected-note{{candidate function not viable: 'this' object is in default address space, but method expects object in address space '__private'}}

  [&] __private {} (); //expected-error{{lambda requires '()' before attribute specifier}} expected-error{{expected body of lambda expression}}

  [&] () mutable __private {} ();
  [&] () __private mutable {} (); //expected-error{{expected body of lambda expression}}
}

