// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -I%S -emit-llvm -o - %s | FileCheck %s

#include <typeinfo>

// CHECK: @_ZTIDn = external constant i8*
int* a = nullptr;

void f() {
  int* a = nullptr;
}

typedef decltype(nullptr) nullptr_t;

nullptr_t get_nullptr();

struct X { };
void g() {
  // CHECK: call i8* @_Z11get_nullptrv()
  int (X::*pmf)(int) = get_nullptr();
}

const std::type_info& f2() {
  return typeid(nullptr_t);
}
