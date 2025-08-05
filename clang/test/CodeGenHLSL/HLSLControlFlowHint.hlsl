// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s

// CHECK: define {{.*}} i32 {{.*}}test_branch{{.*}}(i32 {{.*}} [[VALD:%.*]])
// CHECK: [[PARAM:%.*]] = load i32, ptr [[VALD]].addr, align 4
// CHECK: [[CMP:%.*]] = icmp sgt i32 [[PARAM]], 0
// CHECK: br i1 [[CMP]], label %if.then, label %if.else, !hlsl.controlflow.hint [[HINT_BRANCH:![0-9]+]]
export int test_branch(int X){
    int resp;
    [branch] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: define {{.*}} i32 {{.*}}test_flatten{{.*}}(i32 {{.*}} [[VALD:%.*]])
// CHECK: [[PARAM:%.*]] = load i32, ptr [[VALD]].addr, align 4
// CHECK: [[CMP:%.*]] = icmp sgt i32 [[PARAM]], 0
// CHECK: br i1 [[CMP]], label %if.then, label %if.else, !hlsl.controlflow.hint [[HINT_FLATTEN:![0-9]+]]
export int test_flatten(int X){
    int resp;
    [flatten] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: define {{.*}} i32 {{.*}}test_no_attr{{.*}}(i32 {{.*}} [[VALD:%.*]])
// CHECK-NOT: !hlsl.controlflow.hint
export int test_no_attr(int X){
    int resp;
    if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

//CHECK: [[HINT_BRANCH]] = !{!"hlsl.controlflow.hint", i32 1}
//CHECK: [[HINT_FLATTEN]] = !{!"hlsl.controlflow.hint", i32 2}
