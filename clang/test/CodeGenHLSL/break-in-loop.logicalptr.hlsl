// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -fexperimental-logical-pointer %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-compute -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -fexperimental-logical-pointer %s -o - | FileCheck %s

// Regression test for issue #245605: llvm.structured.alloca was emitted after a
// basic block terminator when a break statement inside a loop triggered a normal
// cleanup destination slot allocation.

// CHECK-LABEL: define {{.*}} @_Z4mainj(
// CHECK: entry:
// CHECK: %{{.*}} = call elementtype(i32) ptr @llvm.structured.alloca.p0()
// CHECK: %{{.*}} = call elementtype(i32) ptr @llvm.structured.alloca.p0()
// CHECK-NOT: llvm.structured.alloca
// CHECK: br label %for.cond
[numthreads(1,1,1)]
void main(uint tid : SV_DispatchThreadID) {
    for (int i = 0; i < 10; i++) {
        if (i == 3) break;
    }
}

// CHECK-LABEL: define {{.*}} @_Z10test_whilei(
// CHECK: entry:
// CHECK: %{{.*}} = call elementtype(i32) ptr @llvm.structured.alloca.p0()
// CHECK-NOT: llvm.structured.alloca
// CHECK: br label %while.cond
export void test_while(int n) {
    while (n > 0) {
        if (n == 5) break;
        n--;
    }
}
