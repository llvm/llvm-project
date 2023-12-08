// RUN: %clang_cc1 -triple i686-windows-itanium -fdeclspec -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-windows-itanium -fdeclspec -fcxx-exceptions -fno-rtti -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-EH-IMPORT

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

void f() {
  throw base();
}

// CHECK-DAG: @_ZTIi = dso_local dllexport constant
// CHECK-DAG: @_ZTSi = dso_local dllexport constant

// CHECK-DAG: @_ZTI7derived = dso_local dllexport constant
// CHECK-DAG: @_ZTS7derived = dso_local dllexport constant
// CHECK-DAG: @_ZTV7derived = dso_local dllexport unnamed_addr constant

// CHECK-DAG: @_ZTI4base = external dllimport constant

// CHECK-EH-IMPORT: @_ZTS4base = linkonce_odr dso_local constant
// CHECK-EH-IMPORT: @_ZTI4base = linkonce_odr dso_local constant

struct __declspec(dllimport) gatekeeper {};
struct zuul : gatekeeper {
  virtual ~zuul();
};
zuul::~zuul() {}

// CHECK-DAG: @_ZTI10gatekeeper = linkonce_odr dso_local constant
// CHECK-DAG: @_ZTS10gatekeeper = linkonce_odr dso_local constant
