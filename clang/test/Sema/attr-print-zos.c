// RUN: %clang_cc1 %s -triple s390x-ibm-zos -ast-print -fzos-extensions | FileCheck %s

// CHECK: int * __ptr32 p32;
int * __ptr32 p32;

// CHECK: char * __ptr32 c32;
char * __ptr32 c32;

// CHECK: void * __ptr32 v32;
void * __ptr32 v32;

// CHECK: int * __ptr32 *q;
int * __ptr32 *q;

// CHECK: void *func(int * __ptr32 p);
void *func(int * __ptr32 p);

// CHECK: int * __ptr32 func1(int * __ptr32 p);
int * __ptr32 func1(int * __ptr32 p);

// CHECK: int *func2(void * __ptr32 p);
int *func2(void * __ptr32 p);

// CHECK: int *const __ptr32 r;
int * __ptr32 const r;

// CHECK: int ** __ptr32 *v;
int * *__ptr32* v;

// CHECK: int *** __ptr32 *z;
int ** * __ptr32 * z;
