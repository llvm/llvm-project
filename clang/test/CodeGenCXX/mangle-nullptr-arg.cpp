// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s

template<int *ip> struct IP {};

// CHECK: define void @_Z5test12IPILPi0EE
void test1(IP<nullptr>) {}

struct X{ };
template<int X::*pm> struct PM {};

// CHECK: define void @_Z5test22PMILM1Xi0EE
void test2(PM<nullptr>) { }

// CHECK: define void @_Z5test316DependentTypePtrIPiLS0_0EE
template<typename T, T x> struct DependentTypePtr {};
void test3(DependentTypePtr<int*,nullptr>) { }
