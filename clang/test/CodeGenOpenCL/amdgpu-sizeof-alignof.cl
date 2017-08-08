// RUN: %clang_cc1 -triple r600 -cl-std=CL1.2 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-mesa-mesa3d -cl-std=CL1.2 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn---opencl -cl-std=CL1.2 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn---opencl -cl-std=CL2.0 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn---amdgizcl -cl-std=CL1.2 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn---amdgizcl -cl-std=CL2.0 %s -emit-llvm -o - | FileCheck %s

#ifdef __AMDGCN__
#define PTSIZE 8
#else
#define PTSIZE 4
#endif

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef global void *global_ptr_t;
typedef constant void *constant_ptr_t;
typedef local void *local_ptr_t;
typedef private void *private_ptr_t;

void check(bool);

void test() {
  // CHECK-NOT: call void @check(i1 zeroext false)
  check(sizeof(size_t) == PTSIZE);
  check(__alignof__(size_t) == PTSIZE);
  check(sizeof(intptr_t) == PTSIZE);
  check(__alignof__(intptr_t) == PTSIZE);
  check(sizeof(uintptr_t) == PTSIZE);
  check(__alignof__(uintptr_t) == PTSIZE);
  check(sizeof(ptrdiff_t) == PTSIZE);
  check(__alignof__(ptrdiff_t) == PTSIZE);

  check(sizeof(char) == 1);
  check(__alignof__(char) == 1);
  check(sizeof(short) == 2);
  check(__alignof__(short) == 2);
  check(sizeof(int) == 4);
  check(__alignof__(int) == 4);
  check(sizeof(long) == 8);
  check(__alignof__(long) == 8);
#ifdef cl_khr_fp16
  check(sizeof(half) == 2);
  check(__alignof__(half) == 2);
#endif
  check(sizeof(float) == 4);
  check(__alignof__(float) == 4);
#ifdef cl_khr_fp64
  check(sizeof(double) == 8);
  check(__alignof__(double) == 8);
#endif

  check(sizeof(void*) == (__OPENCL_C_VERSION__ >= 200 ? 8 : 4));
  check(__alignof__(void*) == (__OPENCL_C_VERSION__ >= 200 ? 8 : 4));
  check(sizeof(global_ptr_t) == PTSIZE);
  check(__alignof__(global_ptr_t) == PTSIZE);
  check(sizeof(constant_ptr_t) == PTSIZE);
  check(__alignof__(constant_ptr_t) == PTSIZE);
  check(sizeof(local_ptr_t) == 4);
  check(__alignof__(private_ptr_t) == 4);
}
