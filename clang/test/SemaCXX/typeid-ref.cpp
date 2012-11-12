// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
namespace std {
  class type_info;
}

struct X { };

void f() {
  // CHECK: @_ZTS1X = linkonce_odr constant
  // CHECK: @_ZTI1X = linkonce_odr unnamed_addr constant 
  (void)typeid(X&);
}
