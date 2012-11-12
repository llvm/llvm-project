// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

template<typename T> void a(T);
template<> void a(int) {}

// CHECK: define void @_Z1aIiEvT_

namespace X {
template<typename T> void b(T);
template<> void b(int) {}
}

// CHECK: define void @_ZN1X1bIiEEvT_
