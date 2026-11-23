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

// GH170243: in C89 a for-init variable belongs to the enclosing scope, so
// leaving the loop must not run its cleanup; in C99 the loop owns it.

// C89-LABEL: define{{.*}} void @test_for_init_break()
// C99-LABEL: define{{.*}} void @test_for_init_break()
void test_for_init_break(void) {
  for (__attribute__((cleanup(cleaner))) int i = 0;;)
    break;
}
// C89:       for.cond:
// C89-NEXT:    br label %for.end
// C89:       for.end:
// C89-NEXT:    call void @cleaner(ptr noundef %i)
// C89-NEXT:    ret void

// C99:       for.cond:
// C99-NEXT:    call void @cleaner(ptr noundef %i)
// C99-NEXT:    br label %for.end
// C99:       for.end:
// C99-NEXT:    ret void

// C89-LABEL: define{{.*}} void @test_for_init_continue(
// C99-LABEL: define{{.*}} void @test_for_init_continue(
void test_for_init_continue(int n) {
  for (__attribute__((cleanup(cleaner))) int i = 0; i < n; i++)
    continue;
}
// C89:         br i1 %cmp, label %for.body, label %for.end
// C89:       for.body:
// C89-NEXT:    br label %for.inc
// C89:       for.end:
// C89-NEXT:    call void @cleaner(ptr noundef %i)
// C89-NEXT:    ret void

// C99:         br i1 %cmp, label %for.body, label %for.cond.cleanup
// C99:       for.cond.cleanup:
// C99-NEXT:    call void @cleaner(ptr noundef %i)
// C99-NEXT:    br label %for.end
// C99:       for.body:
// C99-NEXT:    br label %for.inc
// C99:       for.end:
// C99-NEXT:    ret void

// C89-LABEL: define{{.*}} void @test_for_init_goto()
// C99-LABEL: define{{.*}} void @test_for_init_goto()
void test_for_init_goto(void) {
  for (__attribute__((cleanup(cleaner))) int i = 0;;)
    goto out;
out:
  return;
}
// C89:       for.cond:
// C89-NEXT:    br label %out
// C89:       out:
// C89-NEXT:    call void @cleaner(ptr noundef %i)
// C89-NEXT:    ret void

// C99:       for.cond:
// C99-NEXT:    store i32 {{[0-9]+}}, ptr %cleanup.dest.slot
// C99-NEXT:    call void @cleaner(ptr noundef %i)
// C99-NEXT:    %cleanup.dest = load i32, ptr %cleanup.dest.slot
// C99-NEXT:    switch i32 %cleanup.dest, label %unreachable [
// C99-NEXT:      i32 {{[0-9]+}}, label %out
// C99-NEXT:    ]
// C99:       out:
// C99-NEXT:    ret void

// C89-LABEL: define{{.*}} i32 @test_for_init_return()
// C99-LABEL: define{{.*}} i32 @test_for_init_return()
int test_for_init_return(void) {
  for (__attribute__((cleanup(cleaner))) int i = 1;;)
    return i;
}
// C89:       for.cond:
// C89-NEXT:    %[[V:.*]] = load i32, ptr %i
// C89-NEXT:    call void @cleaner(ptr noundef %i)
// C89-NEXT:    ret i32 %[[V]]

// C99:       for.cond:
// C99-NEXT:    %[[V:.*]] = load i32, ptr %i
// C99-NEXT:    call void @cleaner(ptr noundef %i)
// C99-NEXT:    ret i32 %[[V]]

// C89-LABEL: define{{.*}} void @test_for_init_nested(
// C99-LABEL: define{{.*}} void @test_for_init_nested(
void test_for_init_nested(int n) {
  for (__attribute__((cleanup(cleaner))) int a = 1;; a = 0) {
    for (__attribute__((cleanup(cleaner))) int s = 0; s < n; ++s)
      ;
    if (!a)
      break;
  }
}
// C89:         br i1 %cmp, label %for.body, label %[[INNER_END:for.end[0-9]*]]
// C89:       [[INNER_END]]:
// C89:         br i1 %tobool, label %if.end, label %if.then
// C89:       if.then:
// C89-NEXT:    store i32 {{[0-9]+}}, ptr %cleanup.dest.slot
// C89-NEXT:    br label %cleanup
// C89:       cleanup:
// C89-NEXT:    call void @cleaner(ptr noundef %s)
// C89-NEXT:    %cleanup.dest = load i32, ptr %cleanup.dest.slot
// C89-NEXT:    switch i32 %cleanup.dest, label %unreachable [
// C89-NEXT:      i32 0, label %cleanup.cont
// C89-NEXT:      i32 {{[0-9]+}}, label %[[OUTER_END:for.end[0-9]*]]
// C89-NEXT:    ]
// C89:       [[OUTER_END]]:
// C89-NEXT:    call void @cleaner(ptr noundef %a)
// C89-NEXT:    ret void

// C99:         br i1 %cmp, label %for.body, label %for.cond.cleanup
// C99:       for.cond.cleanup:
// C99-NEXT:    call void @cleaner(ptr noundef %s)
// C99-NEXT:    br label %[[INNER_END:for.end[0-9]*]]
// C99:       [[INNER_END]]:
// C99:         br i1 %tobool, label %if.end, label %if.then
// C99:       if.then:
// C99-NEXT:    call void @cleaner(ptr noundef %a)
// C99-NEXT:    br label %[[OUTER_END:for.end[0-9]*]]
// C99:       [[OUTER_END]]:
// C99-NEXT:    ret void

// C89-LABEL: define{{.*}} void @test_for_init_in_block(
// C99-LABEL: define{{.*}} void @test_for_init_in_block(
void test_for_init_in_block(int n) {
  if (n) {
    for (__attribute__((cleanup(cleaner))) int i = 0;;)
      break;
  }
}
// C89:       for.cond:
// C89-NEXT:    br label %for.end
// C89:       for.end:
// C89-NEXT:    call void @cleaner(ptr noundef %i)
// C89-NEXT:    br label %if.end

// C99:       for.cond:
// C99-NEXT:    call void @cleaner(ptr noundef %i)
// C99-NEXT:    br label %for.end
// C99:       for.end:
// C99-NEXT:    br label %if.end
