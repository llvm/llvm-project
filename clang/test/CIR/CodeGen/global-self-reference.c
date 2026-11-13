// RUN: split-file %s %t

// A global whose initializer refers to itself must resolve the
// self-reference to the CIR type. Warm and cold table cases are
// separate TUs below, with different symbol names.

//--- warm.c

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/warm.c -o %t/warm-cir.ll
// RUN: FileCheck --input-file=%t/warm-cir.ll %t/warm.c -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/warm.c -o %t/warm.ll
// RUN: FileCheck --input-file=%t/warm.ll %t/warm.c -check-prefix=LLVM

// LLVM: @arr = {{.*}}global <{ ptr, ptr, [98 x ptr] }> <{ ptr getelementptr {{.*}} i64 40), ptr getelementptr {{.*}} i64 56),
// LLVM: @later = {{.*}}global ptr getelementptr {{.*}} @arr, i64 56)

// Preceding globals so the module symbol table is cached.
int x;
int *px = &x;
void *arr[100] = {&arr[5], &arr[7], 0};
void **later = &arr[7];

//--- cold.c

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/cold.c -o %t/cold-cir.ll
// RUN: FileCheck --input-file=%t/cold-cir.ll %t/cold.c -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/cold.c -o %t/cold.ll
// RUN: FileCheck --input-file=%t/cold.ll %t/cold.c -check-prefix=LLVM

// LLVM: @solo = {{.*}}global <{ ptr, ptr, [98 x ptr] }> <{ ptr getelementptr {{.*}} i64 40), ptr getelementptr {{.*}} i64 56),
// LLVM: @follower = {{.*}}global ptr getelementptr {{.*}} @solo, i64 56)

// Cold table: the self-referential global is the first symbol in the TU.
void *solo[100] = {&solo[5], &solo[7], 0};
void **follower = &solo[7];
