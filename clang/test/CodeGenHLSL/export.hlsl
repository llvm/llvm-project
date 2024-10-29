// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define void @_Z2f1v() [[Attr:\#[0-9]+]]
export void f1() {
}

// CHECK: define void @_ZN11MyNamespace2f2Ev() [[Attr]]
namespace MyNamespace {
  export void f2() {
  }
}

export {
// CHECK: define void @_Z2f3v() [[Attr]]
// CHECK: define void @_Z2f4v() [[Attr]]
    void f3() {}
    void f4() {}
}

// CHECK: attributes [[Attr]] = { {{.*}} "hlsl.export" {{.*}} }
