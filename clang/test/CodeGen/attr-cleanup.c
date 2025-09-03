// RUN: %clang_cc1 -emit-llvm %s -o %t

void f(void* arg);
void g(void) {
  __attribute__((cleanup(f))) void *g;
}

// Test for cleanup in for-loop initialization (PR #154624)
// RUN: %clang_cc1 -std=c89 -emit-llvm %s -o - | FileCheck %s --check-prefix=C89
// RUN: %clang_cc1 -std=c99 -emit-llvm %s -o - | FileCheck %s --check-prefix=C99

void cleaner(int *p);

// C89-LABEL: define{{.*}} void @test_for_loop_cleanup()  
// C99-LABEL: define{{.*}} void @test_for_loop_cleanup()
void test_for_loop_cleanup(void) {
  for (__attribute__((cleanup(cleaner))) int i = 42; 0;)
    ;
  
#ifndef __STDC_VERSION__
  if (i > 40) {
    // do something
  }
#endif
}

// C89: if.end:
// C89-NEXT: call void @cleaner(ptr noundef %i)
// C89-NEXT: ret void

// C99: for.cond.cleanup:
// C99-NEXT: call void @cleaner(ptr noundef %i)
