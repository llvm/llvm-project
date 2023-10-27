// Tests use-after-return detection and reporting.
// RUN: %clang_hwasan -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// Run the same test as above, but using the __hwasan_add_frame_record libcall.
// The output should be the exact same.
// RUN: %clang_hwasan -g %s -o %t -mllvm -hwasan-record-stack-history=libcall && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// Stack histories currently are not recorded on x86.
// XFAIL: target=x86_64{{.*}}

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
  // CHECK: Cause: stack tag-mismatch
  // CHECK: is located in stack of thread
  // CHECK: Potentially referenced stack objects:
  // CHECK-NEXT: zzz in buggy {{.*}}stack-uar.c:[[@LINE-20]]
  // CHECK-NEXT: Memory tags around the buggy address

  // NOSYM: Previously allocated frames:
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: Memory tags around the buggy address

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
