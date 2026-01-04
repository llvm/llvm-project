// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -fno-builtin -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif

struct __jmp_buf_tag { int n; };
struct __ucontext_t_tag { int n; };
int setjmp(struct __jmp_buf_tag*);
int sigsetjmp(struct __jmp_buf_tag*, int);
int _setjmp(struct __jmp_buf_tag*);
int __sigsetjmp(struct __jmp_buf_tag*, int);
int _setjmpex(struct __jmp_buf_tag* env);
int getcontext(struct __ucontext_t_tag*);

typedef struct __jmp_buf_tag jmp_buf[1];
typedef struct __jmp_buf_tag sigjmp_buf[1];
typedef struct __ucontext_t_tag ucontext_t[1];

#ifdef __cplusplus
}
#endif

void f(void) {
  jmp_buf jb;
  ucontext_t ut;
  // CHECK: call {{.*}}@setjmp(
  setjmp(jb);
  // CHECK: call {{.*}}@sigsetjmp(
  sigsetjmp(jb, 0);
  // CHECK: call {{.*}}@_setjmp(
  _setjmp(jb);
  // CHECK: call {{.*}}@__sigsetjmp(
  __sigsetjmp(jb, 0);
  // CHECK: call {{.*}}@_setjmpex(
  _setjmpex(jb);
  // CHECK: call {{.*}}@getcontext(
  getcontext(ut);
}

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @setjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @sigsetjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @_setjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @__sigsetjmp(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @_setjmpex(

// CHECK: ; Function Attrs: returns_twice
// CHECK-NEXT: declare {{.*}} @getcontext(
