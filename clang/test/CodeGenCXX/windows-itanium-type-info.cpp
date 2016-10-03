// RUN: %clang_cc1 -triple i686-windows-itanium -fdeclspec -emit-llvm %s -o - | FileCheck %s

namespace __cxxabiv1 {
class __declspec(dllexport) __fundamental_type_info {
public:
  virtual ~__fundamental_type_info();
};

__fundamental_type_info::~__fundamental_type_info() {}
}

// CHECK: @_ZTIi = dllexport constant

