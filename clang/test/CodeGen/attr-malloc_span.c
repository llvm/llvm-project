// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s 

int *Mem;

typedef struct {
  void *p;
  int size;
} sized_ptr;

// It should not set the no alias attribute (for now).
__attribute__((malloc_span)) sized_ptr MallocSpan(){ return (sized_ptr){ .p = Mem };}
// CHECK: define dso_local { ptr, i32 } @MallocSpan
