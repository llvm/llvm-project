// Tests use-after-return detection and reporting.
// RUN: %clang_hwasan -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O3 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// Run the same test as above, but using the __hwasan_add_frame_record libcall.
// The output should be the exact same.
// RUN: %clang_hwasan -g %s -o %t -mllvm -hwasan-record-stack-history=libcall && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// Stack histories currently are not recorded on x86.
// XFAIL: target=x86_64{{.*}}

#include <assert.h>
#include <sanitizer/hwasan_interface.h>

void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

__attribute__((noinline))
char *buggy() {
  char zzz[0x800];
  char yyy[0x800];
  // Tags for stack-allocated variables can occasionally be zero, resulting in
  // a false negative for this test. The tag allocation algorithm is not easy
  // to fix, hence we work around it: if the tag is zero, we use the
  // neighboring variable instead, which must have a different (hence non-zero)
  // tag.
  char *volatile p;
  if (__hwasan_tag_pointer(zzz, 0) == zzz) {
    assert(__hwasan_tag_pointer(yyy, 0) != yyy);
    p = yyy;
  } else {
    p = zzz;
  }
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
  // CHECK: Cause: use-after-scope
  // CHECK-NEXT: 0x{{.*}} is located 0 bytes inside a 2048-byte local variable {{zzz|yyy}} [0x{{.*}},0x{{.*}}) in buggy {{.*}}stack-uar.c:
  // CHECK: Memory tags around the buggy address

  // NOSYM: Previously allocated frames:
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uar.c.tmp+0x{{.*}}){{$}}
  // NOSYM: Memory tags around the buggy address

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
