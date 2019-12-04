//RUN: %clang_cc1 %s -cl-std=clc++ -pedantic -ast-dump -verify | FileCheck %s

//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'int (int) const __generic'
auto glambda = [](auto a) { return a; };

__kernel void foo() {
  int i;
//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'void () const __generic'
  auto  llambda = [&]() {i++;};
  llambda();
  glambda(1);
  // Test lambda with default parameters
//CHECK: CXXMethodDecl {{.*}} constexpr operator() 'void () const __generic'
  [&] {i++;} ();
  __constant auto err = [&]() {}; //expected-note-re{{candidate function not viable: address space mismatch in 'this' argument ('__constant (lambda at {{.*}})'), parameter type must be 'const __generic (lambda at {{.*}})'}}
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
