// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefix=GENERIC -check-prefix=COMMON %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa-opencl | FileCheck -check-prefix=AMD -check-prefix=COMMON %s

// Checking for null instead of @__NSConcreteGlobalBlock symbol
// COMMON: @__block_literal_global = internal addrspace(1) constant { i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } { i8** null
void (^block_A)(local void *) = ^(local void *a) {
  return;
};

void foo(){
  int i;
// Checking for null instead of @_NSConcreteStackBlock symbol
// COMMON: store i8* null, i8** %block.isa
  int (^ block_B)(void) = ^{
    return i;
  };
}
