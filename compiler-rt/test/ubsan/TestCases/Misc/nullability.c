// RUN: %clang -w -fsanitize=nullability-arg,nullability-assign,nullability-return %s -O3 -o %t
// RUN: %run %t foo 2>&1 | count 0
// RUN: %run %t 2>&1 | FileCheck %s

// CHECK: nullability.c:[[@LINE+2]]:51: runtime error: null pointer returned from function declared to never return null
// CHECK-NEXT: nullability.c:[[@LINE+1]]:6: note: _Nonnull return type annotation specified here
int *_Nonnull nonnull_retval1(int *p) { return p; }

// CHECK: nullability.c:1001:22: runtime error: null pointer passed as argument 2, which is declared to never be null
// CHECK-NEXT: nullability.c:[[@LINE+1]]:56: note: _Nonnull type annotation specified here
int *_Nonnull nonnull_retval2(int *_Nonnull arg1, int *_Nonnull arg2,
                              int *_Nullable arg3, int *arg4, int arg5, ...) {
  return arg1;
}

// CHECK: nullability.c:1002:15: runtime error: null pointer passed as argument 1, which is declared to never be null
// CHECK-NEXT: nullability.c:[[@LINE+1]]:23: note: _Nonnull type annotation specified here
void nonnull_arg(int *_Nonnull p) {}

void nonnull_assign1(int *p) {
  int *_Nonnull local;
// CHECK: nullability.c:[[@LINE+1]]:9: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
  local = p;
}

void nonnull_assign2(int *p) {
  int *_Nonnull arr[1];
  // CHECK: nullability.c:[[@LINE+1]]:10: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
  arr[0] = p;
}

struct S1 {
  int *_Nonnull mptr;
};

void nonnull_assign3(int *p) {
  struct S1 s;
  // CHECK: nullability.c:[[@LINE+1]]:10: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
  s.mptr = p;
}

// CHECK: nullability.c:[[@LINE+1]]:52: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
void nonnull_init1(int *p) { int *_Nonnull local = p; }

// CHECK: nullability.c:[[@LINE+2]]:53: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
// CHECK: nullability.c:[[@LINE+1]]:56: runtime error: _Nonnull binding to null pointer of type 'int * _Nonnull'
void nonnull_init2(int *p) { int *_Nonnull arr[] = {p, p}; }

int main(int argc, char **argv) {
  int *p = (argc > 1) ? &argc : ((int *)0);

#line 1000
  nonnull_retval1(p);
  nonnull_retval2(p, p, p, p, 0, 0, 0, 0);
  nonnull_arg(p);
  nonnull_assign1(p);
  nonnull_assign2(p);
  nonnull_assign3(p);
  nonnull_init1(p);
  nonnull_init2(p);
  return 0;
}
