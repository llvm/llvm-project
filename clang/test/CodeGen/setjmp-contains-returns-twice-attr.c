// Test that functions containing setjmp / sigsetjmp / _setjmp calls, and
// functions containing calls to user-declared __attribute__((returns_twice))
// functions, are annotated with the contains_returns_twice_call attribute.
//
// The attribute is used by BasicAA to conservatively return MayAlias for
// local allocas in such functions, preventing miscompilation via longjmp
// re-entry paths invisible in the forward CFG.
//
// See: https://github.com/llvm/llvm-project/issues/198967

// RUN: %clang_cc1 -x c %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

struct __jmp_buf_tag { int n; };
typedef struct __jmp_buf_tag jmp_buf[1];
typedef struct __jmp_buf_tag sigjmp_buf[1];

int setjmp(struct __jmp_buf_tag *);
int sigsetjmp(struct __jmp_buf_tag *, int);
int _setjmp(struct __jmp_buf_tag *);

__attribute__((__returns_twice__)) void checkpoint(void);

// CHECK: ; Function Attrs:{{.*}}contains_returns_twice_call
// CHECK-NEXT: define{{.*}}@bar_setjmp(
void bar_setjmp(void) {
  jmp_buf buf;
  setjmp(buf);
}

// CHECK: ; Function Attrs:{{.*}}contains_returns_twice_call
// CHECK-NEXT: define{{.*}}@bar_sigsetjmp(
void bar_sigsetjmp(void) {
  sigjmp_buf buf;
  sigsetjmp(buf, 0);
}

// CHECK: ; Function Attrs:{{.*}}contains_returns_twice_call
// CHECK-NEXT: define{{.*}}@bar_setjmp_underscore(
void bar_setjmp_underscore(void) {
  jmp_buf buf;
  _setjmp(buf);
}

// A user-declared returns_twice function must also trigger the attribute.
// CHECK: ; Function Attrs:{{.*}}contains_returns_twice_call
// CHECK-NEXT: define{{.*}}@bar_checkpoint(
void bar_checkpoint(void) {
  checkpoint();
}

// A function with no returns_twice call must NOT have the attribute.
// The emitted "; Function Attrs:" line for bar_plain will contain only the
// default function attributes (noinline nounwind optnone), not
// contains_returns_twice_call.
// CHECK: ; Function Attrs: noinline nounwind optnone
// CHECK-NEXT: define{{.*}}@bar_plain(
void bar_plain(void) {
  int x = 42;
  (void)x;
}
