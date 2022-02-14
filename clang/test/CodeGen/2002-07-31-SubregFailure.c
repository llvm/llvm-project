// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


typedef union {
   long (*ap)[4];
} ptrs;

void DoAssignIteration(void) {
  ptrs abase;
  abase.ap+=27;
  Assignment(*abase.ap);
}


