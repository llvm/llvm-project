// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define void @_Z2f1v() [[Attr:\#[0-9]+]]
export void f1() {
}

// CHECK: define void @_ZN11MyNamespace2f2Ev()
namespace MyNamespace {
  export void f2() {
  }
}

export {
// CHECK: define void @_Z2f3v()
// CHECK: define void @_Z2f4v()
    void f3() {}
    void f4() {}
}