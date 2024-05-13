// Tests use-after-scope detection and reporting.
// RUN: %clang_hwasan -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O2 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O2 -g %s -DBUFFER_SIZE=1000000 -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O2 -g %s -DBUFFER_SIZE=2000000 -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// RUN: %clang_hwasan -mllvm -hwasan-use-after-scope=false -g %s -o %t && %run %t 2>&1

// RUN: %clang_hwasan -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Run the same test as above, but using the __hwasan_add_frame_record libcall.
// The output should be the exact same.
// RUN: %clang_hwasan -mllvm -hwasan-record-stack-history=libcall -g %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// Stack histories currently are not recorded on x86.
// XFAIL: target=x86_64{{.*}}

#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>

#ifndef BUFFER_SIZE
#  define BUFFER_SIZE 0x800
#endif

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
    char zzz[BUFFER_SIZE] = {};
    char yyy[BUFFER_SIZE] = {};
    // With -hwasan-generate-tags-with-calls=false, stack tags can occasionally
    // be zero, leading to a false negative
    // (https://github.com/llvm/llvm-project/issues/69221). Work around it by
    // using the neighboring variable, which is guaranteed by
    // -hwasan-generate-tags-with-calls=false to have a different (hence
    // non-zero) tag.
    if (__hwasan_tag_pointer(zzz, 0) == zzz) {
      assert(__hwasan_tag_pointer(yyy, 0) != yyy);
      p = yyy;
    } else {
      p = zzz;
    }
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
  // CHECK: Cause: use-after-scope
  // CHECK-NEXT: 0x{{.*}} is located 0 bytes inside a {{.*}}-byte local variable {{zzz|yyy}} [0x{{.*}}) in buggy {{.*}}stack-uas.c:
  // CHECK: Memory tags around the buggy address

  // NOSYM: Previously allocated frames:
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM-NEXT: record_addr:0x{{.*}} record:0x{{.*}} ({{.*}}/stack-uas.c.tmp+0x{{.*}}){{$}}
  // NOSYM: Memory tags around the buggy address

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in buggy
}
