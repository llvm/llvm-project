// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fclangir \
// RUN:   -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefixes=CHECK,LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefixes=CHECK,OGCG --input-file=%t.ll
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fclangir \
// RUN:   -O3 -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa \
// RUN:   -O3 -emit-llvm %s -o /dev/null

// OGCG-DAG: constant %struct.BoolArray { [3 x i8] c"\01\00\01" }
// OGCG-DAG: constant [3 x i8] c"\01\00\00"

struct BoolArray {
  bool values[3];
};

// CHECK-LABEL: define {{.+}}i1 @record()
// LLVM: store %struct.BoolArray { [3 x i8] c"\01\00\01" }
// CHECK: ret i1
extern "C" bool record() {
  BoolArray value{true, false, true};
  return value.values[0];
}

// CHECK-LABEL: define {{.+}}i1 @partial_array()
// LLVM: store [3 x i8] c"\01\00\00"
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
