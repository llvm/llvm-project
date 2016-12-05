// RUN: %clang_cc1 -triple i686-windows-itanium -fdeclspec -emit-llvm %s -o - | FileCheck %s

namespace __cxxabiv1 {
class __declspec(dllexport) __fundamental_type_info {
public:
  virtual ~__fundamental_type_info();
};

__fundamental_type_info::~__fundamental_type_info() {}
}

struct __declspec(dllimport) base {
  virtual void method();
};
struct __declspec(dllexport) derived : base {
  virtual ~derived();
};
derived::~derived() {
  method();
}

// CHECK-DAG: @_ZTIi = dllexport constant
// CHECK-DAG: @_ZTSi = dllexport constant

// CHECK-DAG: @_ZTI7derived = dllexport constant
// CHECK-DAG: @_ZTS7derived = dllexport constant
// CHECK-DAG: @_ZTV7derived = dllexport unnamed_addr constant

// CHECK-DAG: @_ZTI4base = external dllimport constant
// CHECK-DAG: @_ZTS4base = external dllimport constant
// CHECK-NOT: @_ZTV4base = external dllimport constant

