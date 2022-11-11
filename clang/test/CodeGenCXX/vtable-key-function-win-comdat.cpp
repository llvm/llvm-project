// RUN: %clang_cc1 %s -triple=x86_64-pc-windows-gnu -emit-llvm -o - | FileCheck %s

namespace std { class type_info; }
extern void use(const std::type_info &rtti);

struct Test1a {
  Test1a();
  virtual void foo();
  virtual void bar();
};

// V-table needs to be defined weakly.
Test1a::Test1a() { use(typeid(Test1a)); }
// This defines the key function.
inline void Test1a::foo() {}

// CHECK:     $_ZTV6Test1a = comdat any
// CHECK:     $_ZTS6Test1a = comdat any
// CHECK:     $_ZTI6Test1a = comdat any
// CHECK-NOT: $_ZTS6Test1a.1 = comdat any
// CHECK-NOT: $_ZTI6Test1a.1 = comdat any

// CHECK: @_ZTV6Test1a = linkonce_odr dso_local unnamed_addr constant {{.*}} ptr @_ZTI6Test1a
// CHECK: @_ZTS6Test1a = linkonce_odr dso_local constant
// CHECK: @_ZTI6Test1a = linkonce_odr dso_local constant {{.*}} ptr @_ZTS6Test1a
