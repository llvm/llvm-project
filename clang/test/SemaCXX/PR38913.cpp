// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s %}

// PR38913
// Check that we instantiate attributes on declarations for...

// ...a member class of a class template specialization
template<class T> struct A { struct __attribute__((abi_tag("ATAG"))) X { }; };
A<int>::X* a() { return 0; } // CHECK-DAG: @_Z1aB4ATAGv

// ...a member class template
template<class T> struct B { template<class U> struct __attribute__((abi_tag("BTAG"))) X { }; };
B<int>::X<int>* b() { return 0; } // CHECK-DAG: @_Z1bB4BTAGv
