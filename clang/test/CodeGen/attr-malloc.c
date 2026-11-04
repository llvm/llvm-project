// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s 

int *Mem;
void dealloc(int*);

__attribute__((malloc)) int *MallocFunc(){ return Mem;}
// CHECK: define[[BEFORE:.*]] noalias[[AFTER:.*]]@MallocFunc
// Ensure these two do not generate noalias here.
__attribute__((malloc(dealloc))) int *MallocFunc2(){ return Mem;}
// CHECK: define[[BEFORE]][[AFTER]]@MallocFunc2
__attribute__((malloc(dealloc, 1))) int *MallocFunc3(){ return Mem;}
// CHECK: define[[BEFORE]][[AFTER]]@MallocFunc3
