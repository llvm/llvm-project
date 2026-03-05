// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=WRAP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=NOWRAP

#define N 42

typedef struct {
  const char* ptr;
} A;

// WRAP:   cir.binop(sub, {{.*}}, {{.*}}) : !s32i
// NOWRAP: cir.binop(sub, {{.*}}, {{.*}}) nsw : !s32i
void foo(int* ar, int len) {
  int x = ar[len - N];
}

// check that the ptr_stride is generated in both cases (i.e. no NYI fails)

// WRAP:    cir.ptr_stride
// NOWRAP:  cir.ptr_stride
void bar(A* a, unsigned n) {
  a->ptr = a->ptr + n;
}

// WRAP    cir.ptr_stride
// NOWRAP: cir.ptr_stride
void baz(A* a) {
  a->ptr--;
}


