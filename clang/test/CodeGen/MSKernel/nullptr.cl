// On amdgcn nullptr is -1. We don't rewrite AST in such case
// RUN: %clang_cc1 -fms-kernel -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple amdgcn -emit-llvm -o - | FileCheck %s

typedef struct {
  private char *p1;
  local char *p2;
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy1;

typedef struct {
  constant char *p3;
  global char *p4;
  generic char *p5;
} StructTy2;


// CHECK: @fold_int5 ={{.*}} local_unnamed_addr addrspace(1) global i32 3, align 4
int fold_int5 = (int) &((private StructTy1*)0)->p2;

// CHECK: @fold_int5_local ={{.*}} local_unnamed_addr addrspace(1) global i32 3, align 4
int fold_int5_local = (int) &((local StructTy1*)0)->p2;


