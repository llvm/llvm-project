// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -stack-protector 2 -emit-llvm -o - | FileCheck %s

class A {
 public:
  virtual ~A() {}
};

A g;

// CHECK: define internal void @__cxx_global_var_init() [[ATTR0:#[0-9]+]]
// CHECK: define internal void @_GLOBAL__sub_I_funcattrs_global_ctor_dtor.cpp() [[ATTR0]]
// CHECK: attributes [[ATTR0]] = {{{.*}} sspstrong {{.*}}}
