// RUN: %clang_cc1 -fnamed-loops -triple x86_64-unknown-linux -std=c++20 -emit-llvm -o - %s | FileCheck %s

static int a[10]{};
struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

bool g(int);
bool h();

// CHECK-LABEL: define {{.*}} void @_Z2f1v()
// CHECK: entry:
// CHECK:   %__range1 = alloca ptr, align 8
// CHECK:   %__begin1 = alloca ptr, align 8
// CHECK:   %__end1 = alloca ptr, align 8
// CHECK:   %i = alloca i32, align 4
// CHECK:   br label %x
// CHECK: x:
// CHECK:   store ptr @_ZL1a, ptr %__range1, align 8
// CHECK:   store ptr @_ZL1a, ptr %__begin1, align 8
// CHECK:   store ptr getelementptr inbounds (i32, ptr @_ZL1a, i64 10), ptr %__end1, align 8
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   %0 = load ptr, ptr %__begin1, align 8
// CHECK:   %1 = load ptr, ptr %__end1, align 8
// CHECK:   %cmp = icmp ne ptr %0, %1
// CHECK:   br i1 %cmp, label %for.body, label %for.end
// CHECK: for.body:
// CHECK:   %2 = load ptr, ptr %__begin1, align 8
// CHECK:   %3 = load i32, ptr %2, align 4
// CHECK:   store i32 %3, ptr %i, align 4
// CHECK:   %4 = load i32, ptr %i, align 4
// CHECK:   %call = call {{.*}} i1 @_Z1gi(i32 {{.*}} %4)
// CHECK:   br i1 %call, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   br label %for.end
// CHECK: if.end:
// CHECK:   %5 = load i32, ptr %i, align 4
// CHECK:   %call1 = call {{.*}} i1 @_Z1gi(i32 {{.*}} %5)
// CHECK:   br i1 %call1, label %if.then2, label %if.end3
// CHECK: if.then2:
// CHECK:   br label %for.inc
// CHECK: if.end3:
// CHECK:   br label %for.inc
// CHECK: for.inc:
// CHECK:   %6 = load ptr, ptr %__begin1, align 8
// CHECK:   %incdec.ptr = getelementptr inbounds nuw i32, ptr %6, i32 1
// CHECK:   store ptr %incdec.ptr, ptr %__begin1, align 8
// CHECK:   br label %for.cond
// CHECK: for.end:
// CHECK:   ret void
void f1() {
  x: for (int i : a) {
    if (g(i)) break x;
    if (g(i)) continue x;
  }
}

// CHECK-LABEL: define {{.*}} void @_Z2f2v()
// CHECK: entry:
// CHECK:   %n1 = alloca %struct.NonTrivialDestructor, align 1
// CHECK:   %__range2 = alloca ptr, align 8
// CHECK:   %__begin2 = alloca ptr, align 8
// CHECK:   %__end2 = alloca ptr, align 8
// CHECK:   %i = alloca i32, align 4
// CHECK:   %n2 = alloca %struct.NonTrivialDestructor, align 1
// CHECK:   %cleanup.dest.slot = alloca i32, align 4
// CHECK:   %n3 = alloca %struct.NonTrivialDestructor, align 1
// CHECK:   %n4 = alloca %struct.NonTrivialDestructor, align 1
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @_Z1gi(i32 {{.*}} 0)
// CHECK:   br i1 %call, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   store ptr @_ZL1a, ptr %__range2, align 8
// CHECK:   store ptr @_ZL1a, ptr %__begin2, align 8
// CHECK:   store ptr getelementptr inbounds (i32, ptr @_ZL1a, i64 10), ptr %__end2, align 8
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   %0 = load ptr, ptr %__begin2, align 8
// CHECK:   %1 = load ptr, ptr %__end2, align 8
// CHECK:   %cmp = icmp ne ptr %0, %1
// CHECK:   br i1 %cmp, label %for.body, label %for.end
// CHECK: for.body:
// CHECK:   %2 = load ptr, ptr %__begin2, align 8
// CHECK:   %3 = load i32, ptr %2, align 4
// CHECK:   store i32 %3, ptr %i, align 4
// CHECK:   %4 = load i32, ptr %i, align 4
// CHECK:   %call1 = call {{.*}} i1 @_Z1gi(i32 {{.*}} %4)
// CHECK:   br i1 %call1, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   store i32 4, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup
// CHECK: if.end:
// CHECK:   %5 = load i32, ptr %i, align 4
// CHECK:   %call2 = call {{.*}} i1 @_Z1gi(i32 {{.*}} %5)
// CHECK:   br i1 %call2, label %if.then3, label %if.end4
// CHECK: if.then3:
// CHECK:   store i32 3, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup
// CHECK: if.end4:
// CHECK:   %6 = load i32, ptr %i, align 4
// CHECK:   %call5 = call {{.*}} i1 @_Z1gi(i32 {{.*}} %6)
// CHECK:   br i1 %call5, label %if.then6, label %if.end7
// CHECK: if.then6:
// CHECK:   store i32 6, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup
// CHECK: if.end7:
// CHECK:   %7 = load i32, ptr %i, align 4
// CHECK:   %call8 = call {{.*}} i1 @_Z1gi(i32 {{.*}} %7)
// CHECK:   br i1 %call8, label %if.then9, label %if.end10
// CHECK: if.then9:
// CHECK:   store i32 7, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup
// CHECK: if.end10:
// CHECK:   call void @_ZN20NonTrivialDestructorD1Ev(ptr {{.*}} %n3)
// CHECK:   store i32 0, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup
// CHECK: cleanup:
// CHECK:   call void @_ZN20NonTrivialDestructorD1Ev(ptr {{.*}} %n2)
// CHECK:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
// CHECK:   switch i32 %cleanup.dest, label %cleanup11 [
// CHECK:     i32 0, label %cleanup.cont
// CHECK:     i32 6, label %for.end
// CHECK:     i32 7, label %for.inc
// CHECK:   ]
// CHECK: cleanup.cont:
// CHECK:   br label %for.inc
// CHECK: for.inc:
// CHECK:   %8 = load ptr, ptr %__begin2, align 8
// CHECK:   %incdec.ptr = getelementptr inbounds nuw i32, ptr %8, i32 1
// CHECK:   store ptr %incdec.ptr, ptr %__begin2, align 8
// CHECK:   br label %for.cond
// CHECK: for.end:
// CHECK:   call void @_ZN20NonTrivialDestructorD1Ev(ptr {{.*}} %n4)
// CHECK:   store i32 0, ptr %cleanup.dest.slot, align 4
// CHECK:   br label %cleanup11
// CHECK: cleanup11:
// CHECK:   call void @_ZN20NonTrivialDestructorD1Ev(ptr {{.*}} %n1)
// CHECK:   %cleanup.dest12 = load i32, ptr %cleanup.dest.slot, align 4
// CHECK:   switch i32 %cleanup.dest12, label %unreachable [
// CHECK:     i32 0, label %cleanup.cont13
// CHECK:     i32 4, label %while.end
// CHECK:     i32 3, label %while.cond
// CHECK:   ]
// CHECK: cleanup.cont13:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
// CHECK: unreachable:
// CHECK:   unreachable
void f2() {
  l1: while (g(0)) {
    NonTrivialDestructor n1;
    l2: for (int i : a) {
      NonTrivialDestructor n2;
      if (g(i)) break l1;
      if (g(i)) continue l1;
      if (g(i)) break l2;
      if (g(i)) continue l2;
      NonTrivialDestructor n3;
    }
    NonTrivialDestructor n4;
  }
}

template <bool Continue>
void f3() {
  l1: while (g(1)) {
    for (;g(2);) {
      if constexpr (Continue) continue l1;
      else break l1;
    }
  }
}

// CHECK-LABEL: define {{.*}} void @_Z2f3ILb1EEvv()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @_Z1gi(i32 {{.*}} 1)
// CHECK:   br i1 %call, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   %call1 = call {{.*}} i1 @_Z1gi(i32 {{.*}} 2)
// CHECK:   br i1 %call1, label %for.body, label %for.end
// CHECK: for.body:
// CHECK:   br label %while.cond
// CHECK: for.end:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
template void f3<true>();

// CHECK-LABEL: define {{.*}} void @_Z2f3ILb0EEvv()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @_Z1gi(i32 {{.*}} 1)
// CHECK:   br i1 %call, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   %call1 = call {{.*}} i1 @_Z1gi(i32 {{.*}} 2)
// CHECK:   br i1 %call1, label %for.body, label %for.end
// CHECK: for.body:
// CHECK:   br label %while.end
// CHECK: for.end:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
template void f3<false>();
