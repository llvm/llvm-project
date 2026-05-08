// RUN: %clang_cc1 -triple x86_64-unknown-linux -std=c23 -fdefer-ts -emit-llvm %s -o - -O1 -disable-llvm-passes | FileCheck %s

// Test that cleanups emitted in a '_Defer' don't clobber the cleanup slot; we
// test this using lifetime intrinsics, which are emitted starting at -O1.

void g();

// CHECK-LABEL: define {{.*}} void @f1()
// CHECK: entry:
// CHECK-NEXT:   %i = alloca i32, align 4
// CHECK-NEXT:   %cleanup.dest.slot = alloca i32, align 4
// CHECK-NEXT:   %j = alloca i32, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %i)
// CHECK-NEXT:   store i32 0, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: for.cond:
// CHECK-NEXT:   %0 = load i32, ptr %i, align 4
// CHECK-NEXT:   %cmp = icmp eq i32 %0, 1
// CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
// CHECK: if.then:
// CHECK-NEXT:   store i32 2, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup
// CHECK: if.end:
// CHECK-NEXT:   store i32 0, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup
// CHECK: cleanup:
// CHECK-NEXT:   %cleanup.dest.saved = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %j)
// CHECK-NEXT:   store i32 0, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: for.cond1:
// CHECK-NEXT:   %1 = load i32, ptr %j, align 4
// CHECK-NEXT:   %cmp2 = icmp ne i32 %1, 1
// CHECK-NEXT:   br i1 %cmp2, label %for.body, label %for.cond.cleanup
// CHECK: for.cond.cleanup:
// CHECK-NEXT:   store i32 5, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %j)
// CHECK-NEXT:   br label %for.end
// CHECK: for.body:
// CHECK-NEXT:   call void @g()
// CHECK-NEXT:   br label %for.inc
// CHECK: for.inc:
// CHECK-NEXT:   %2 = load i32, ptr %j, align 4
// CHECK-NEXT:   %inc = add nsw i32 %2, 1
// CHECK-NEXT:   store i32 %inc, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: for.end:
// CHECK-NEXT:   store i32 %cleanup.dest.saved, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest, label %cleanup6 [
// CHECK-NEXT:     i32 0, label %cleanup.cont
// CHECK-NEXT:   ]
// CHECK: cleanup.cont:
// CHECK-NEXT:   br label %for.inc4
// CHECK: for.inc4:
// CHECK-NEXT:   %3 = load i32, ptr %i, align 4
// CHECK-NEXT:   %inc5 = add nsw i32 %3, 1
// CHECK-NEXT:   store i32 %inc5, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: cleanup6:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:   br label %for.end7
// CHECK: for.end7:
// CHECK-NEXT:   ret void
void f1() {
  for (int i = 0;; i++) {
    _Defer {
      for (int j = 0; j != 1; j++) {
        g();
      }
    }
    if (i == 1) break;
  }
}

// CHECK-LABEL: define {{.*}} void @f2()
// CHECK: entry:
// CHECK-NEXT:   %i = alloca i32, align 4
// CHECK-NEXT:   %cleanup.dest.slot = alloca i32, align 4
// CHECK-NEXT:   %j = alloca i32, align 4
// CHECK-NEXT:   %k = alloca i32, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %i)
// CHECK-NEXT:   store i32 0, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: for.cond:
// CHECK-NEXT:   %0 = load i32, ptr %i, align 4
// CHECK-NEXT:   %cmp = icmp eq i32 %0, 1
// CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
// CHECK: if.then:
// CHECK-NEXT:   store i32 2, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup
// CHECK: if.end:
// CHECK-NEXT:   store i32 0, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup
// CHECK: cleanup:
// CHECK-NEXT:   %cleanup.dest.saved = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %j)
// CHECK-NEXT:   store i32 0, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: for.cond1:
// CHECK-NEXT:   %1 = load i32, ptr %j, align 4
// CHECK-NEXT:   %cmp2 = icmp eq i32 %1, 1
// CHECK-NEXT:   br i1 %cmp2, label %if.then3, label %if.end4
// CHECK: if.then3:
// CHECK-NEXT:   store i32 5, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup5
// CHECK: if.end4:
// CHECK-NEXT:   store i32 0, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   br label %cleanup5
// CHECK: cleanup5:
// CHECK-NEXT:   %cleanup.dest.saved6 = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %k)
// CHECK-NEXT:   store i32 0, ptr %k, align 4
// CHECK-NEXT:   br label %for.cond7
// CHECK: for.cond7:
// CHECK-NEXT:   %2 = load i32, ptr %k, align 4
// CHECK-NEXT:   %cmp8 = icmp ne i32 %2, 1
// CHECK-NEXT:   br i1 %cmp8, label %for.body, label %for.cond.cleanup
// CHECK: for.cond.cleanup:
// CHECK-NEXT:   store i32 8, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %k)
// CHECK-NEXT:   br label %for.end
// CHECK: for.body:
// CHECK-NEXT:   call void @g()
// CHECK-NEXT:   br label %for.inc
// CHECK: for.inc:
// CHECK-NEXT:   %3 = load i32, ptr %k, align 4
// CHECK-NEXT:   %inc = add nsw i32 %3, 1
// CHECK-NEXT:   store i32 %inc, ptr %k, align 4
// CHECK-NEXT:   br label %for.cond7
// CHECK: for.end:
// CHECK-NEXT:   store i32 %cleanup.dest.saved6, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest, label %cleanup12 [
// CHECK-NEXT:     i32 0, label %cleanup.cont
// CHECK-NEXT:   ]
// CHECK: cleanup.cont:
// CHECK-NEXT:   br label %for.inc10
// CHECK: for.inc10:
// CHECK-NEXT:   %4 = load i32, ptr %j, align 4
// CHECK-NEXT:   %inc11 = add nsw i32 %4, 1
// CHECK-NEXT:   store i32 %inc11, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: cleanup12:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %j)
// CHECK-NEXT:   br label %for.end13
// CHECK: for.end13:
// CHECK-NEXT:   store i32 %cleanup.dest.saved, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   %cleanup.dest14 = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest14, label %cleanup18 [
// CHECK-NEXT:     i32 0, label %cleanup.cont15
// CHECK-NEXT:   ]
// CHECK: cleanup.cont15:
// CHECK-NEXT:   br label %for.inc16
// CHECK: for.inc16:
// CHECK-NEXT:   %5 = load i32, ptr %i, align 4
// CHECK-NEXT:   %inc17 = add nsw i32 %5, 1
// CHECK-NEXT:   store i32 %inc17, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: cleanup18:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:   br label %for.end19
// CHECK: for.end19:
// CHECK-NEXT:   ret void
void f2() {
  for (int i = 0;; i++) {
    _Defer {
      for (int j = 0;; j++) {
        _Defer {
          for (int k = 0; k != 1; k++) {
            g();
          }
        }
	if (j == 1) break;
      }
    }
    if (i == 1) break;
  }
}
