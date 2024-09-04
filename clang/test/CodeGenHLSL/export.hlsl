// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define void @"?f1@@YAXXZ"() [[Attr:\#[0-9]+]]
export void f1() {
}

// CHECK: define void @"?f2@MyNamespace@@YAXXZ"() [[Attr]]
namespace MyNamespace {
  export void f2() {
  }
}

export {
// CHECK: define void @"?f3@@YAXXZ"() [[Attr]]
// CHECK: define void @"?f4@@YAXXZ"() [[Attr]]
    void f3() {}
    void f4() {}
}

// CHECK: attributes [[Attr]] = { {{.*}} "hlsl.export" {{.*}} }
