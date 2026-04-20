// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR      --input-file=%t.cir %s
// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -fno-builtin -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR      --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ %s -triple x86_64-linux-gnu -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR      --input-file=%t.cir %s

// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -fno-builtin -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -x c++ %s -triple x86_64-linux-gnu -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t.ll %s
// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -fno-builtin -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ %s -triple x86_64-linux-gnu -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM      --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=LLVM-DECL --input-file=%t.ll %s

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

// The wildcard in the attributes is to allow this to work with and without -fno-builtin.

// CIR: cir.func private @setjmp({{.*}}) -> !s32i attributes {{{.*}}returns_twice}
// CIR: cir.func private @sigsetjmp({{.*}}) -> !s32i attributes {{{.*}}returns_twice}
// CIR: cir.func private @_setjmp({{.*}}) -> !s32i attributes {{{.*}}returns_twice}
// CIR: cir.func private @__sigsetjmp({{.*}}) -> !s32i attributes {{{.*}}returns_twice}
// CIR: cir.func private @_setjmpex({{.*}}) -> !s32i attributes {{{.*}}returns_twice}
// CIR: cir.func private @getcontext({{.*}}) -> !s32i attributes {{{.*}}returns_twice}

void f(void) {
  jmp_buf jb;
  ucontext_t ut;
  // CIR: cir.call @setjmp(
  // LLVM: call {{.*}} @setjmp(
  setjmp(jb);
  // CIR: cir.call @sigsetjmp(
  // LLVM: call {{.*}} @sigsetjmp(
  sigsetjmp(jb, 0);
  // CIR: cir.call @_setjmp(
  // LLVM: call {{.*}} @_setjmp(
  _setjmp(jb);
  // CIR: cir.call @__sigsetjmp(
  // LLVM: call {{.*}} @__sigsetjmp(
  __sigsetjmp(jb, 0);
  // CIR: cir.call @_setjmpex(
  // LLVM: call {{.*}} @_setjmpex(
  _setjmpex(jb);
  // CIR: cir.call @getcontext(
  // LLVM: call {{.*}} @getcontext(
  getcontext(ut);
}

// These are checked with a separate RUN and check prefix because classic
// codegen emits them after the definition of @f, while CIR emits them before.

// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @setjmp(
// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @sigsetjmp(
// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @_setjmp(
// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @__sigsetjmp(
// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @_setjmpex(
// LLVM-DECL: ; Function Attrs: returns_twice
// LLVM-DECL-NEXT: declare {{.*}} @getcontext(
