// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct S {
  int i;
  int j;
};

int f() {
  auto [i, j] = S{1, 42};
  return [&i, j] {
    return i + j;
  }();
}

// Ensures the representation of the lambda, the order of the
// 1st 2nd don't matter except for ABI-esque things, but make sure
// that the ref-capture is a ptr, and 'j' is captured by value.
// CHECK: %[[LAMBDA_TY:.+]] = type <{ ptr, i32, [4 x i8] }>

// Check the captures themselves.
// CHECK: define{{.*}} i32 @_Z1fv()
// CHECK: %[[BINDING:.+]] = alloca %struct.S
// CHECK: %[[LAMBDA:.+]] = alloca %[[LAMBDA_TY]]

// Copy a pointer to the binding, for reference capture.
// CHECK: %[[LAMBDA_CAP_PTR:.+]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[LAMBDA]], i32 0, i32 0
// CHECK: %[[BINDING_PTR:.+]] = getelementptr inbounds nuw %struct.S, ptr %[[BINDING]], i32 0, i32 0
// CHECK: store ptr %[[BINDING_PTR]], ptr %[[LAMBDA_CAP_PTR]]

// Copy the integer from the binding, for copy capture.
// CHECK: %[[LAMBDA_CAP_INT:.+]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[LAMBDA]], i32 0, i32 1
// CHECK: %[[PTR_TO_J:.+]] = getelementptr inbounds nuw %struct.S, ptr %[[BINDING]], i32 0, i32 1
// CHECK: %[[J_COPY:.+]] = load i32, ptr %[[PTR_TO_J]]
// CHECK: store i32 %[[J_COPY]], ptr %[[LAMBDA_CAP_INT]]

// Ensure the captures are properly extracted in operator().
// CHECK: define{{.*}} i32 @"_ZZ1fvENK3$_0clEv"
// CHECK: %[[THIS_ADDR:.+]] = alloca ptr
// CHECK: %[[THIS_PTR:.+]] = load ptr, ptr %[[THIS_ADDR]]

// Load 'i', passed by reference.
// CHECK: %[[LAMBDA_GEP_TO_PTR:.+]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[THIS_PTR]], i32 0, i32 0
// CHECK: %[[I_PTR:.+]] = load ptr, ptr %[[LAMBDA_GEP_TO_PTR]]
// CHECK: %[[I_VALUE:.+]] = load i32, ptr %[[I_PTR]]

// Load the 'j', passed by value.
// CHECK: %[[LAMBDA_GEP_TO_INT:.+]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[THIS_PTR]], i32 0, i32 1
// CHECK: %[[J_VALUE:.+]] = load i32, ptr %[[LAMBDA_GEP_TO_INT]]
