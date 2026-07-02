// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=MAP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=NOCC

void f(void);
void *alloc(unsigned long);
void release(void *);
__attribute__((returns_twice)) int returns_twice(void);

struct cleanup_ctx {
  int fd0;
  int fd1;
};

#define CLEANUP_CTX_MACRO(ctx, close0, close1) \
  {                                            \
    if (close0)                                \
      f();                                     \
    if (close1)                                \
      f();                                     \
    (ctx)->fd0 = -1;                           \
    (ctx)->fd1 = -1;                           \
    release((void *)(ctx));                    \
    ctx = (struct cleanup_ctx *)0;             \
  }

int after_call(void) {
  f();
  return 1;
}

int setjmp_like(void) {
  if (returns_twice() == 0)
    return 1;
  return 2;
}

int after_guard_and_call(int ret) {
  void *child = alloc(4);
  if (!child)
    return 2;
  if (ret) {
    f();
    return 1;
  }
  return 0;
}

int cleanup_macro_after_call(int ret) {
  struct cleanup_ctx *child = (struct cleanup_ctx *)alloc(sizeof(*child));
  struct cleanup_ctx *parent = (struct cleanup_ctx *)alloc(sizeof(*parent));
  if (!child || !parent)
    return 2;
  if (ret) {
    f();
    CLEANUP_CTX_MACRO(child, 1, 1);
    CLEANUP_CTX_MACRO(parent, 1, 1);
    return 1;
  }
  return 0;
}

// MAP-LABEL: after_call:
// MAP: Gap,File 0, [[CALL_LINE:[0-9]+]]:7 -> [[RET_LINE:[0-9]+]]:3 = #1
// MAP: File 0, [[RET_LINE]]:3 -> [[END_LINE:[0-9]+]]:2 = #1
// MAP-LABEL: setjmp_like:
// MAP: Branch,File 0, [[COND_LINE:[0-9]+]]:7 -> [[COND_LINE]]:27 = #1, (#2 - #1)
// MAP-LABEL: after_guard_and_call:
// MAP: Gap,File 0, [[CALL_LINE:[0-9]+]]:9 -> [[RET_LINE:[0-9]+]]:5 = #{{[0-9]+}}
// MAP-LABEL: cleanup_macro_after_call:
// MAP: Expansion,File 0, [[FIRST_CLEANUP:[0-9]+]]:5 -> [[FIRST_CLEANUP]]:22 = #{{[0-9]+}}
// MAP: Expansion,File 0, [[SECOND_CLEANUP:[0-9]+]]:5 -> [[SECOND_CLEANUP]]:22 = #{{[0-9]+}}
// NOCC-LABEL: setjmp_like:
// NOCC: Branch,File 0, [[COND_LINE:[0-9]+]]:7 -> [[COND_LINE]]:27 = #1, (#0 - #1)
