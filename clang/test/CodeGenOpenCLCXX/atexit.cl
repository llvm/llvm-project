//RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -emit-llvm -O0 -o - | FileCheck %s

struct S {
  ~S(){};
};
S s;

//CHECK-LABEL: define internal spir_func void @__cxx_global_var_init()
//CHECK: call spir_func i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.S addrspace(4)*)* @_ZNU3AS41SD1Ev to void (i8*)*), i8* null, i8 addrspace(1)* @__dso_handle)

//CHECK: declare spir_func i32 @__cxa_atexit(void (i8*)*, i8*, i8 addrspace(1)*)
