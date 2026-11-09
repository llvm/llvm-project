// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fclangir \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fclangir \
// RUN:   -O3 -emit-llvm %s -o /dev/null

struct BoolArray {
  bool values[3];
};

// CHECK-LABEL: define {{.+}}i1 @record()
// CHECK: store %struct.BoolArray { [3 x i8] c"\01\00\01" }
// CHECK: ret i1
extern "C" bool record() {
  BoolArray value{true, false, true};
  return value.values[0];
}

// CHECK-LABEL: define {{.+}}i1 @partial_array()
// CHECK: store [3 x i8] c"\01\00\00"
// CHECK: ret i1
extern "C" bool partial_array() {
  bool values[3] = {true};
  return values[2];
}

// CHECK-LABEL: define {{.+}}i1 @scalar_bool()
// CHECK: ret i1
extern "C" bool scalar_bool() {
  return true;
}
