// RUN: %clang_cc1 -triple x86_64-unknown-linux -std=c23 -fdefer-ts -emit-llvm %s -o - -O1 -disable-llvm-passes | FileCheck %s

// Test that cleanups emitted in a 'defer' don't clobber the cleanup slot; we
// test this using lifetime intrinsics, which are emitted starting at -O1.
//
// Note that the IR below contains fewer cleanup slots than one might intuitively
// expect because some of them are optimised out (we just emit a direct branch
// instead if the cleanup slot would only be written to once); the important part
// is that we don't clobber a cleanup slot while executing its cleanup.

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
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %j)
// CHECK-NEXT:   store i32 0, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: for.cond1:
// CHECK-NEXT:   %1 = load i32, ptr %j, align 4
// CHECK-NEXT:   %cmp2 = icmp ne i32 %1, 1
// CHECK-NEXT:   br i1 %cmp2, label %for.body, label %for.cond.cleanup
// CHECK: for.cond.cleanup:
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
// CHECK-NEXT:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest, label %cleanup7 [
// CHECK-NEXT:     i32 0, label %cleanup.cont
// CHECK-NEXT:   ]
// CHECK: cleanup.cont:
// CHECK-NEXT:   br label %for.inc5
// CHECK: for.inc5:
// CHECK-NEXT:   %3 = load i32, ptr %i, align 4
// CHECK-NEXT:   %inc6 = add nsw i32 %3, 1
// CHECK-NEXT:   store i32 %inc6, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: cleanup7:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:   br label %for.end8
// CHECK: for.end8:
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
// CHECK-NEXT:   %cleanup.dest.slot4 = alloca i32, align 4
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
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %j)
// CHECK-NEXT:   store i32 0, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: for.cond1:
// CHECK-NEXT:   %1 = load i32, ptr %j, align 4
// CHECK-NEXT:   %cmp2 = icmp eq i32 %1, 1
// CHECK-NEXT:   br i1 %cmp2, label %if.then3, label %if.end5
// CHECK: if.then3:
// CHECK-NEXT:   store i32 5, ptr %cleanup.dest.slot4, align 4
// CHECK-NEXT:   br label %cleanup6
// CHECK: if.end5:
// CHECK-NEXT:   store i32 0, ptr %cleanup.dest.slot4, align 4
// CHECK-NEXT:   br label %cleanup6
// CHECK: cleanup6:
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %k)
// CHECK-NEXT:   store i32 0, ptr %k, align 4
// CHECK-NEXT:   br label %for.cond7
// CHECK: for.cond7:
// CHECK-NEXT:   %2 = load i32, ptr %k, align 4
// CHECK-NEXT:   %cmp8 = icmp ne i32 %2, 1
// CHECK-NEXT:   br i1 %cmp8, label %for.body, label %for.cond.cleanup
// CHECK: for.cond.cleanup:
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
// CHECK-NEXT:   %cleanup.dest = load i32, ptr %cleanup.dest.slot4, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest, label %cleanup13 [
// CHECK-NEXT:     i32 0, label %cleanup.cont
// CHECK-NEXT:   ]
// CHECK: cleanup.cont:
// CHECK-NEXT:   br label %for.inc11
// CHECK: for.inc11:
// CHECK-NEXT:   %4 = load i32, ptr %j, align 4
// CHECK-NEXT:   %inc12 = add nsw i32 %4, 1
// CHECK-NEXT:   store i32 %inc12, ptr %j, align 4
// CHECK-NEXT:   br label %for.cond1
// CHECK: cleanup13:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %j)
// CHECK-NEXT:   br label %for.end14
// CHECK: for.end14:
// CHECK-NEXT:   %cleanup.dest15 = load i32, ptr %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest15, label %cleanup19 [
// CHECK-NEXT:     i32 0, label %cleanup.cont16
// CHECK-NEXT:   ]
// CHECK: cleanup.cont16:
// CHECK-NEXT:   br label %for.inc17
// CHECK: for.inc17:
// CHECK-NEXT:   %5 = load i32, ptr %i, align 4
// CHECK-NEXT:   %inc18 = add nsw i32 %5, 1
// CHECK-NEXT:   store i32 %inc18, ptr %i, align 4
// CHECK-NEXT:   br label %for.cond
// CHECK: cleanup19:
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %i)
// CHECK-NEXT:   br label %for.end20
// CHECK: for.end20:
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
