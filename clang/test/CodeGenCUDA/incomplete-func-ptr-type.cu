// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -x hip %s -o - \
// RUN: | FileCheck %s

#define __global__ __attribute__((global))
// CHECK: @_Z4kern7TempValIjE = constant ptr @_Z19__device_stub__kern7TempValIjE, align 8
// CHECK: @0 = private unnamed_addr constant [19 x i8] c"_Z4kern7TempValIjE\00", align 1
template <typename type>
struct TempVal {
  type value;
};

__global__ void kern(TempVal<unsigned int> in_val);

int main(int argc, char ** argv) {
  auto* fptr = &(kern);
// CHECK:   store ptr @_Z4kern7TempValIjE, ptr %fptr, align 8
  return 0;
}
// CHECK:  define dso_local void @_Z19__device_stub__kern7TempValIjE(i32 %in_val.coerce) #1 {
// CHECK:  %2 = call i32 @hipLaunchByPtr(ptr @_Z4kern7TempValIjE)

// CHECK:  define internal void @__hip_register_globals(ptr %0) {
// CHECK:    %1 = call i32 @__hipRegisterFunction(ptr %0, ptr @_Z4kern7TempValIjE, ptr @0, ptr @0, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)

__global__ void kern(TempVal<unsigned int> in_val) {
}

