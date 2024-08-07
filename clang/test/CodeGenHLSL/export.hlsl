// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang --driver-mode=dxc -T lib_6_3 -Od %s | FileCheck %s --check-prefix=CHECK-MD

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

// CHECK-MD: !dx.exports = !{[[Exp1:![0-9]+]], [[Exp2:![0-9]+]], [[Exp3:![0-9]+]], [[Exp4:![0-9]+]]}
// CHECK-MD: [[Exp1]] = !{ptr @"?f1@@YAXXZ", !"?f1@@YAXXZ"}
// CHECK-MD: [[Exp2]] = !{ptr @"?f2@MyNamespace@@YAXXZ", !"?f2@MyNamespace@@YAXXZ"}
// CHECK-MD: [[Exp3]] = !{ptr @"?f3@@YAXXZ", !"?f3@@YAXXZ"}
// CHECK-MD: [[Exp4]] = !{ptr @"?f4@@YAXXZ", !"?f4@@YAXXZ"}

