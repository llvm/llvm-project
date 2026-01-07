// RUN: %clang_cc1 -std=c89 -emit-llvm %s -o - | FileCheck %s --check-prefix=C89
// RUN: %clang_cc1 -std=c99 -emit-llvm %s -o - | FileCheck %s --check-prefix=C99

void f(void* arg);
void g(void) {
  __attribute__((cleanup(f))) void *g;
}

void cleaner(int *p);

// C89-LABEL: define{{.*}} void @test_nested_for_loop_cleanup()
// C99-LABEL: define{{.*}} void @test_nested_for_loop_cleanup()
void test_nested_for_loop_cleanup(void) {
  for (int i = 10; 0;) {
    for (__attribute__((cleanup(cleaner))) int j = 20; 0;)
      ;
    i = 5; // Some operation after inner loop
  }
}

// C89: for.end:
// C89-NEXT: store i32 5, ptr %i, align 4
// C89-NEXT: call void @cleaner(ptr noundef %j)

// C99: for.cond.cleanup:
// C99-NEXT: call void @cleaner(ptr noundef %j)
// C99-NEXT: br label %for.end
