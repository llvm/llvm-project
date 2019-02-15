// Tests use-after-return detection and reporting.
// RUN: %clang_hwasan -O0 -fno-discard-value-names %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O0 -fno-discard-value-names %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// REQUIRES: stable-runtime

void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

__attribute__((noinline))
char *buggy() {
  char zzz[0x1000];
  char *volatile p = zzz;
  return p;
}

__attribute__((noinline)) void Unrelated1() { int A[2]; USE(&A[0]); }
__attribute__((noinline)) void Unrelated2() { int BB[3]; USE(&BB[0]); }
__attribute__((noinline)) void Unrelated3() { int CCC[4]; USE(&CCC[0]); }

int main() {
  char *p = buggy();
  Unrelated1();
  Unrelated2();
  Unrelated3();
  return *p;
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main{{.*}}stack-uar.c:[[@LINE-2]]
  // CHECK: is located in stack of thread
  // CHECK: Previously allocated frames:
  // CHECK: Unrelated3
  // CHECK: 16 CCC
  // CHECK: Unrelated2
  // CHECK: 12 BB
  // CHECK: Unrelated1
  // CHECK: 8 A
  // CHECK: buggy
  // CHECK: 4096 zzz

  // NOSYM: Previously allocated frames:
  // NOSYM-NEXT: sp: 0x{{.*}} #0 0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: 16 CCC;

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
