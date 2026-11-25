// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c89 -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,C89
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c99 -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,C99
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -std=c++17 -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,C99

// GH170243: before C99 a variable declared in a for-init-statement belongs to
// the enclosing scope, so leaving the loop must not end its lifetime. In C99
// and C++ the loop owns it and its lifetime ends on every exit path.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cplusplus
void *bar(int);

// C89-LABEL: define{{.*}} ptr @foo()
void *foo(void) {
  char *c = "abc";
  for (int a = 1;; a = 0) {
    for (char *s = c; *s; ++s) {
    }
    if (!a)
      break;
  }
}
// C89:         br i1 %tobool, label %for.body, label %[[INNER_END:for.end[0-9]*]]
// C89:       [[INNER_END]]:
// C89:         br i1 %tobool{{[0-9]*}}, label %if.end, label %if.then
// C89:       if.then:
// C89-NEXT:    store i32 {{[0-9]+}}, ptr %cleanup.dest.slot
// C89-NEXT:    br label %cleanup
// C89:       cleanup:
// C89-NEXT:    call void @llvm.lifetime.end.p0(ptr %s)
// C89-NEXT:    %cleanup.dest = load i32, ptr %cleanup.dest.slot
// C89-NEXT:    switch i32 %cleanup.dest, label %unreachable [
// C89-NEXT:      i32 0, label %cleanup.cont
// C89-NEXT:      i32 {{[0-9]+}}, label %[[OUTER_END:for.end[0-9]*]]
// C89-NEXT:    ]
// C89:       [[OUTER_END]]:
// C89-NEXT:    call void @llvm.lifetime.end.p0(ptr %a)
// C89-NEXT:    call void @llvm.lifetime.end.p0(ptr %c)
// C89:         ret ptr

// C89-LABEL: define{{.*}} i32 @main()
int main(void) {
  for (int i;;)
    break;
}
// C89:         call void @llvm.lifetime.start.p0(ptr %i)
// C89-NEXT:    br label %for.cond
// C89:       for.cond:
// C89-NEXT:    br label %for.end
// C89:       for.end:
// C89-NEXT:    call void @llvm.lifetime.end.p0(ptr %i)
// C89:         ret i32
#endif

// CHECK-LABEL: define{{.*}} void @test_break()
void test_break(void) {
  for (int i = 0;;)
    break;
}
// CHECK:         call void @llvm.lifetime.start.p0(ptr %i)
// CHECK-NEXT:    store i32 0, ptr %i
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// C89-NEXT:      br label %for.end
// C89:         for.end:
// C89-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// C89-NEXT:      ret void
// C99-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// C99-NEXT:      br label %for.end
// C99:         for.end:
// C99-NEXT:      ret void

// CHECK-LABEL: define{{.*}} void @test_continue(
void test_continue(int n) {
  for (int i = 0; i < n; i++)
    continue;
}
// C89:           br i1 %cmp, label %for.body, label %for.end
// C99:           br i1 %cmp, label %for.body, label %for.cond.cleanup
// C99:         for.cond.cleanup:
// C99-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// C99-NEXT:      br label %for.end
// CHECK:       for.body:
// CHECK-NEXT:    br label %for.inc
// CHECK:       for.end:
// C89-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:    ret void

// CHECK-LABEL: define{{.*}} void @test_goto()
void test_goto(void) {
  for (int i = 0;;)
    goto out;
out:
  return;
}
// CHECK:       for.cond:
// C89-NEXT:      br label %out
// C89:         out:
// C89-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// C89-NEXT:      ret void
// C99-NEXT:      store i32 {{[0-9]+}}, ptr %cleanup.dest.slot
// C99-NEXT:      call void @llvm.lifetime.end.p0(ptr %i)
// C99-NEXT:      %cleanup.dest = load i32, ptr %cleanup.dest.slot
// C99-NEXT:      switch i32 %cleanup.dest, label %unreachable [
// C99-NEXT:        i32 {{[0-9]+}}, label %out
// C99-NEXT:      ]
// C99:         out:
// C99-NEXT:      ret void

// CHECK-LABEL: define{{.*}} i32 @test_return()
int test_return(void) {
  for (int i = 1;;)
    return i;
}
// CHECK:       for.cond:
// CHECK-NEXT:    %[[V:.*]] = load i32, ptr %i
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:    ret i32 %[[V]]

// CHECK-LABEL: define{{.*}} void @test_nested(
void test_nested(int n) {
  for (int a = 1;; a = 0) {
    for (int s = 0; s < n; ++s)
      ;
    if (!a)
      break;
  }
}
// C89:           br i1 %cmp, label %for.body, label %[[INNER_END:for.end[0-9]*]]
// C89:         [[INNER_END]]:
// C89:           br i1 %tobool, label %if.end, label %if.then
// C89:         if.then:
// C89-NEXT:      store i32 {{[0-9]+}}, ptr %cleanup.dest.slot
// C89-NEXT:      br label %cleanup
// C89:         cleanup:
// C89-NEXT:      call void @llvm.lifetime.end.p0(ptr %s)
// C89-NEXT:      %cleanup.dest = load i32, ptr %cleanup.dest.slot
// C89-NEXT:      switch i32 %cleanup.dest, label %unreachable [
// C89-NEXT:        i32 0, label %cleanup.cont
// C89-NEXT:        i32 {{[0-9]+}}, label %[[OUTER_END:for.end[0-9]*]]
// C89-NEXT:      ]
// C89:         [[OUTER_END]]:
// C89-NEXT:      call void @llvm.lifetime.end.p0(ptr %a)
// C89-NEXT:      ret void

// C99:           br i1 %cmp, label %for.body, label %for.cond.cleanup
// C99:         for.cond.cleanup:
// C99-NEXT:      call void @llvm.lifetime.end.p0(ptr %s)
// C99-NEXT:      br label %[[INNER_END:for.end[0-9]*]]
// C99:         [[INNER_END]]:
// C99:           br i1 %tobool, label %if.end, label %if.then
// C99:         if.then:
// C99-NEXT:      call void @llvm.lifetime.end.p0(ptr %a)
// C99-NEXT:      br label %[[OUTER_END:for.end[0-9]*]]
// C99:         [[OUTER_END]]:
// C99-NEXT:      ret void

#ifdef __cplusplus
}
#endif
