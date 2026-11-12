// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


extern int A[10];
void Func(int *B) { 
  B - &A[5]; 
}

