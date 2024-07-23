// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define void @"?f1@@YAXXZ"()
export void f1() {
}

// CHECK: define void @"?f2@MyNamespace@@YAXXZ"()
namespace MyNamespace {
  export void f2() {
  }
}

export {
// CHECK: define void @"?f3@@YAXXZ"()
// CHECK: define void @"?f4@@YAXXZ"()    
    void f3() {}
    void f4() {}
}
