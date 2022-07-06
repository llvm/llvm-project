// Tests use-after-scope detection and reporting.
// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope=false -g %s -o %t && %run %t 2>&1
// Use after scope is turned off by default.
// RUN: %clang_hwasan -g %s -o %t && %run %t 2>&1

// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Run the same test as above, but using the __hwasan_add_frame_record libcall.
// The output should be the exact same.
// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope -mllvm -hwasan-record-stack-history=libcall -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// REQUIRES: stable-runtime

// Stack histories currently are not recorded on x86.
// XFAIL: x86_64

void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__(""
                       :
                       : "r"(x)
                       : "memory");
}

__attribute__((noinline)) void Unrelated1() {
  int A[2];
  USE(&A[0]);
}
__attribute__((noinline)) void Unrelated2() {
  int BB[3];
  USE(&BB[0]);
}
__attribute__((noinline)) void Unrelated3() {
  int CCC[4];
  USE(&CCC[0]);
}

__attribute__((noinline)) char buggy() {
  char *volatile p;
  {
    char zzz[0x1000];
    p = zzz;
  }
  return *p;
}

int main() {
  Unrelated1();
  Unrelated2();
  Unrelated3();
  char p = buggy();
  return p;
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in buggy{{.*}}stack-uas.c:[[@LINE-10]]
  // CHECK: Cause: stack tag-mismatch
  // CHECK: is located in stack of thread
  // CHECK: Potentially referenced stack objects:
  // CHECK-NEXT: zzz in buggy {{.*}}stack-uas.c:[[@LINE-17]]
  // CHECK-NEXT: Memory tags around the buggy address

  // NOSYM: Previously allocated frames:
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: Memory tags around the buggy address

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in buggy
}
