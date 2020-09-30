// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif

struct __jmp_buf_tag { int n; };
int setjmp(struct __jmp_buf_tag*);
int sigsetjmp(struct __jmp_buf_tag*, int);
int _setjmp(struct __jmp_buf_tag*);
int __sigsetjmp(struct __jmp_buf_tag*, int);

typedef struct __jmp_buf_tag jmp_buf[1];
typedef struct __jmp_buf_tag sigjmp_buf[1];

#ifdef __cplusplus
}
#endif

void f() {
  jmp_buf jb;
  // CHECK: call {{.*}}@setjmp(
  setjmp(jb);
  // CHECK: call {{.*}}@sigsetjmp(
  sigsetjmp(jb, 0);
  // CHECK: call {{.*}}@_setjmp(
  _setjmp(jb);
  // CHECK: call {{.*}}@__sigsetjmp(
  __sigsetjmp(jb, 0);
}

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @setjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @sigsetjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @_setjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @__sigsetjmp(

