// RUN: %clang_cc1 -std=c2y -triple x86_64-apple-darwin -Wno-objc-root-class -emit-llvm -o - %s | FileCheck %s

int g(id x);

// CHECK-LABEL: define void @f1(ptr {{.*}} %y)
// CHECK: entry:
// CHECK:   %y.addr = alloca ptr, align 8
// CHECK:   %x1 = alloca ptr, align 8
// CHECK:   %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
// CHECK:   %items.ptr = alloca [16 x ptr], align 8
// CHECK:   store ptr %y, ptr %y.addr, align 8
// CHECK:   br label %x
// CHECK: x:
// CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
// CHECK:   %0 = load ptr, ptr %y.addr, align 8
// CHECK:   %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
// CHECK:   %call = call i64 @objc_msgSend(ptr {{.*}} %0, ptr {{.*}} %1, ptr {{.*}} %state.ptr, ptr {{.*}} %items.ptr, i64 {{.*}} 16)
// CHECK:   %iszero = icmp eq i64 %call, 0
// CHECK:   br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit
// CHECK: forcoll.loopinit:
// CHECK:   %mutationsptr.ptr = getelementptr inbounds nuw %struct.__objcFastEnumerationState, ptr %state.ptr, i32 0, i32 2
// CHECK:   %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
// CHECK:   %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
// CHECK:   br label %forcoll.loopbody
// CHECK: forcoll.loopbody:
// CHECK:   %forcoll.index = phi i64 [ 0, %forcoll.loopinit ], [ %6, %forcoll.next ], [ 0, %forcoll.refetch ]
// CHECK:   %forcoll.count = phi i64 [ %call, %forcoll.loopinit ], [ %forcoll.count, %forcoll.next ], [ %call8, %forcoll.refetch ]
// CHECK:   %mutationsptr2 = load ptr, ptr %mutationsptr.ptr, align 8
// CHECK:   %statemutations = load i64, ptr %mutationsptr2, align 8
// CHECK:   %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
// CHECK:   br i1 %2, label %forcoll.notmutated, label %forcoll.mutated
// CHECK: forcoll.mutated:
// CHECK:   call void @objc_enumerationMutation(ptr {{.*}} %0)
// CHECK:   br label %forcoll.notmutated
// CHECK: forcoll.notmutated:
// CHECK:   %stateitems.ptr = getelementptr inbounds nuw %struct.__objcFastEnumerationState, ptr %state.ptr, i32 0, i32 1
// CHECK:   %stateitems = load ptr, ptr %stateitems.ptr, align 8
// CHECK:   %currentitem.ptr = getelementptr inbounds ptr, ptr %stateitems, i64 %forcoll.index
// CHECK:   %3 = load ptr, ptr %currentitem.ptr, align 8
// CHECK:   store ptr %3, ptr %x1, align 8
// CHECK:   %4 = load ptr, ptr %x1, align 8
// CHECK:   %call3 = call i32 @g(ptr {{.*}} %4)
// CHECK:   %tobool = icmp ne i32 %call3, 0
// CHECK:   br i1 %tobool, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   br label %forcoll.end
// CHECK: if.end:
// CHECK:   %5 = load ptr, ptr %x1, align 8
// CHECK:   %call4 = call i32 @g(ptr {{.*}} %5)
// CHECK:   %tobool5 = icmp ne i32 %call4, 0
// CHECK:   br i1 %tobool5, label %if.then6, label %if.end7
// CHECK: if.then6:
// CHECK:   br label %forcoll.next
// CHECK: if.end7:
// CHECK:   br label %forcoll.next
// CHECK: forcoll.next:
// CHECK:   %6 = add nuw i64 %forcoll.index, 1
// CHECK:   %7 = icmp ult i64 %6, %forcoll.count
// CHECK:   br i1 %7, label %forcoll.loopbody, label %forcoll.refetch
// CHECK: forcoll.refetch:
// CHECK:   %8 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
// CHECK:   %call8 = call i64 @objc_msgSend(ptr {{.*}} %0, ptr {{.*}} %8, ptr {{.*}} %state.ptr, ptr {{.*}} %items.ptr, i64 {{.*}} 16)
// CHECK:   %9 = icmp eq i64 %call8, 0
// CHECK:   br i1 %9, label %forcoll.empty, label %forcoll.loopbody
// CHECK: forcoll.empty:
// CHECK:   br label %forcoll.end
// CHECK: forcoll.end:
// CHECK:   ret void
void f1(id y) {
  x: for (id x in y) {
    if (g(x)) break x;
    if (g(x)) continue x;
  }
}

// CHECK-LABEL: define void @f2(ptr {{.*}} %y)
// CHECK: entry:
// CHECK:   %y.addr = alloca ptr, align 8
// CHECK:   %x = alloca ptr, align 8
// CHECK:   %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
// CHECK:   %items.ptr = alloca [16 x ptr], align 8
// CHECK:   store ptr %y, ptr %y.addr, align 8
// CHECK:   br label %a
// CHECK: a:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %0 = load ptr, ptr %y.addr, align 8
// CHECK:   %call = call i32 @g(ptr {{.*}} %0)
// CHECK:   %tobool = icmp ne i32 %call, 0
// CHECK:   br i1 %tobool, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   br label %b
// CHECK: b:
// CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
// CHECK:   %1 = load ptr, ptr %y.addr, align 8
// CHECK:   %2 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
// CHECK:   %call1 = call i64 @objc_msgSend(ptr {{.*}} %1, ptr {{.*}} %2, ptr {{.*}} %state.ptr, ptr {{.*}} %items.ptr, i64 {{.*}} 16)
// CHECK:   %iszero = icmp eq i64 %call1, 0
// CHECK:   br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit
// CHECK: forcoll.loopinit:
// CHECK:   %mutationsptr.ptr = getelementptr inbounds nuw %struct.__objcFastEnumerationState, ptr %state.ptr, i32 0, i32 2
// CHECK:   %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
// CHECK:   %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
// CHECK:   br label %forcoll.loopbody
// CHECK: forcoll.loopbody:
// CHECK:   %forcoll.index = phi i64 [ 0, %forcoll.loopinit ], [ %9, %forcoll.next ], [ 0, %forcoll.refetch ]
// CHECK:   %forcoll.count = phi i64 [ %call1, %forcoll.loopinit ], [ %forcoll.count, %forcoll.next ], [ %call17, %forcoll.refetch ]
// CHECK:   %mutationsptr2 = load ptr, ptr %mutationsptr.ptr, align 8
// CHECK:   %statemutations = load i64, ptr %mutationsptr2, align 8
// CHECK:   %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
// CHECK:   br i1 %3, label %forcoll.notmutated, label %forcoll.mutated
// CHECK: forcoll.mutated:
// CHECK:   call void @objc_enumerationMutation(ptr {{.*}} %1)
// CHECK:   br label %forcoll.notmutated
// CHECK: forcoll.notmutated:
// CHECK:   %stateitems.ptr = getelementptr inbounds nuw %struct.__objcFastEnumerationState, ptr %state.ptr, i32 0, i32 1
// CHECK:   %stateitems = load ptr, ptr %stateitems.ptr, align 8
// CHECK:   %currentitem.ptr = getelementptr inbounds ptr, ptr %stateitems, i64 %forcoll.index
// CHECK:   %4 = load ptr, ptr %currentitem.ptr, align 8
// CHECK:   store ptr %4, ptr %x, align 8
// CHECK:   %5 = load ptr, ptr %x, align 8
// CHECK:   %call3 = call i32 @g(ptr {{.*}} %5)
// CHECK:   %tobool4 = icmp ne i32 %call3, 0
// CHECK:   br i1 %tobool4, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   br label %while.end
// CHECK: if.end:
// CHECK:   %6 = load ptr, ptr %x, align 8
// CHECK:   %call5 = call i32 @g(ptr {{.*}} %6)
// CHECK:   %tobool6 = icmp ne i32 %call5, 0
// CHECK:   br i1 %tobool6, label %if.then7, label %if.end8
// CHECK: if.then7:
// CHECK:   br label %while.cond
// CHECK: if.end8:
// CHECK:   %7 = load ptr, ptr %x, align 8
// CHECK:   %call9 = call i32 @g(ptr {{.*}} %7)
// CHECK:   %tobool10 = icmp ne i32 %call9, 0
// CHECK:   br i1 %tobool10, label %if.then11, label %if.end12
// CHECK: if.then11:
// CHECK:   br label %forcoll.end
// CHECK: if.end12:
// CHECK:   %8 = load ptr, ptr %x, align 8
// CHECK:   %call13 = call i32 @g(ptr {{.*}} %8)
// CHECK:   %tobool14 = icmp ne i32 %call13, 0
// CHECK:   br i1 %tobool14, label %if.then15, label %if.end16
// CHECK: if.then15:
// CHECK:   br label %forcoll.next
// CHECK: if.end16:
// CHECK:   br label %forcoll.next
// CHECK: forcoll.next:
// CHECK:   %9 = add nuw i64 %forcoll.index, 1
// CHECK:   %10 = icmp ult i64 %9, %forcoll.count
// CHECK:   br i1 %10, label %forcoll.loopbody, label %forcoll.refetch
// CHECK: forcoll.refetch:
// CHECK:   %11 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
// CHECK:   %call17 = call i64 @objc_msgSend(ptr {{.*}} %1, ptr {{.*}} %11, ptr {{.*}} %state.ptr, ptr {{.*}} %items.ptr, i64 {{.*}} 16)
// CHECK:   %12 = icmp eq i64 %call17, 0
// CHECK:   br i1 %12, label %forcoll.empty, label %forcoll.loopbody
// CHECK: forcoll.empty:
// CHECK:   br label %forcoll.end
// CHECK: forcoll.end:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
void f2(id y) {
  a: while (g(y)) {
    b: for (id x in y) {
      if (g(x)) break a;
      if (g(x)) continue a;
      if (g(x)) break b;
      if (g(x)) continue b;
    }
  }
}
