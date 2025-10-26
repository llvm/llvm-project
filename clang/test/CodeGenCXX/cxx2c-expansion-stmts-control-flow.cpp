// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void h(int, int);

void break_continue() {
  template for (auto x : {1, 2}) {
    break;
    h(1, x);
  }

  template for (auto x : {3, 4}) {
    continue;
    h(2, x);
  }

  template for (auto x : {5, 6}) {
    if (x == 2) break;
    h(3, x);
  }

  template for (auto x : {7, 8}) {
    if (x == 2) continue;
    h(4, x);
  }
}

int break_continue_nested() {
  int sum = 0;

  template for (auto x : {1, 2}) {
    template for (auto y : {3, 4}) {
      if (x == 2) break;
      sum += y;
    }
    sum += x;
  }

  template for (auto x : {5, 6}) {
    template for (auto y : {7, 8}) {
      if (x == 6) continue;
      sum += y;
    }
    sum += x;
  }

  return sum;
}

void label() {
  // Only local labels are allowed in expansion statements.
  template for (auto x : {1, 2, 3}) {
    __label__ a;
    if (x == 1) goto a;
    h(1, x);
    a:;
  }
}


// CHECK-LABEL: define {{.*}} void @_Z14break_continuev()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x11 = alloca i32, align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 3, ptr %x1, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 4, ptr %x2, align 4
// CHECK-NEXT:   br label %expand.end3
// CHECK: expand.end3:
// CHECK-NEXT:   store i32 5, ptr %x4, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x4, align 4
// CHECK-NEXT:   %cmp = icmp eq i32 %0, 2
// CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
// CHECK: if.then:
// CHECK-NEXT:   br label %expand.end10
// CHECK: if.end:
// CHECK-NEXT:   %1 = load i32, ptr %x4, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 3, i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   store i32 6, ptr %x6, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %cmp7 = icmp eq i32 %2, 2
// CHECK-NEXT:   br i1 %cmp7, label %if.then8, label %if.end9
// CHECK: if.then8:
// CHECK-NEXT:   br label %expand.end10
// CHECK: if.end9:
// CHECK-NEXT:   %3 = load i32, ptr %x6, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 3, i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end10
// CHECK: expand.end10:
// CHECK-NEXT:   store i32 7, ptr %x11, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x11, align 4
// CHECK-NEXT:   %cmp12 = icmp eq i32 %4, 2
// CHECK-NEXT:   br i1 %cmp12, label %if.then13, label %if.end14
// CHECK: if.then13:
// CHECK-NEXT:   br label %expand.next15
// CHECK: if.end14:
// CHECK-NEXT:   %5 = load i32, ptr %x11, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 4, i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   store i32 8, ptr %x16, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x16, align 4
// CHECK-NEXT:   %cmp17 = icmp eq i32 %6, 2
// CHECK-NEXT:   br i1 %cmp17, label %if.then18, label %if.end19
// CHECK: if.then18:
// CHECK-NEXT:   br label %expand.end20
// CHECK: if.end19:
// CHECK-NEXT:   %7 = load i32, ptr %x16, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 4, i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end20
// CHECK: expand.end20:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} i32 @_Z21break_continue_nestedv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %y1 = alloca i32, align 4
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   %y9 = alloca i32, align 4
// CHECK-NEXT:   %y15 = alloca i32, align 4
// CHECK-NEXT:   %x23 = alloca i32, align 4
// CHECK-NEXT:   %y24 = alloca i32, align 4
// CHECK-NEXT:   %y30 = alloca i32, align 4
// CHECK-NEXT:   %x38 = alloca i32, align 4
// CHECK-NEXT:   %y39 = alloca i32, align 4
// CHECK-NEXT:   %y45 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   store i32 3, ptr %y, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   %cmp = icmp eq i32 %0, 2
// CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
// CHECK: if.then:
// CHECK-NEXT:   br label %expand.end
// CHECK: if.end:
// CHECK-NEXT:   %1 = load i32, ptr %y, align 4
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %2, %1
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 4, ptr %y1, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x, align 4
// CHECK-NEXT:   %cmp2 = icmp eq i32 %3, 2
// CHECK-NEXT:   br i1 %cmp2, label %if.then3, label %if.end4
// CHECK: if.then3:
// CHECK-NEXT:   br label %expand.end
// CHECK: if.end4:
// CHECK-NEXT:   %4 = load i32, ptr %y1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %6 = load i32, ptr %x, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add6 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add6, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next7
// CHECK: expand.next7:
// CHECK-NEXT:   store i32 2, ptr %x8, align 4
// CHECK-NEXT:   store i32 3, ptr %y9, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x8, align 4
// CHECK-NEXT:   %cmp10 = icmp eq i32 %8, 2
// CHECK-NEXT:   br i1 %cmp10, label %if.then11, label %if.end12
// CHECK: if.then11:
// CHECK-NEXT:   br label %expand.end20
// CHECK: if.end12:
// CHECK-NEXT:   %9 = load i32, ptr %y9, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %10, %9
// CHECK-NEXT:   store i32 %add13, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next14
// CHECK: expand.next14:
// CHECK-NEXT:   store i32 4, ptr %y15, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x8, align 4
// CHECK-NEXT:   %cmp16 = icmp eq i32 %11, 2
// CHECK-NEXT:   br i1 %cmp16, label %if.then17, label %if.end18
// CHECK: if.then17:
// CHECK-NEXT:   br label %expand.end20
// CHECK: if.end18:
// CHECK-NEXT:   %12 = load i32, ptr %y15, align 4
// CHECK-NEXT:   %13 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add19 = add nsw i32 %13, %12
// CHECK-NEXT:   store i32 %add19, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end20
// CHECK: expand.end20:
// CHECK-NEXT:   %14 = load i32, ptr %x8, align 4
// CHECK-NEXT:   %15 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add21 = add nsw i32 %15, %14
// CHECK-NEXT:   store i32 %add21, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   store i32 5, ptr %x23, align 4
// CHECK-NEXT:   store i32 7, ptr %y24, align 4
// CHECK-NEXT:   %16 = load i32, ptr %x23, align 4
// CHECK-NEXT:   %cmp25 = icmp eq i32 %16, 6
// CHECK-NEXT:   br i1 %cmp25, label %if.then26, label %if.end27
// CHECK: if.then26:
// CHECK-NEXT:   br label %expand.next29
// CHECK: if.end27:
// CHECK-NEXT:   %17 = load i32, ptr %y24, align 4
// CHECK-NEXT:   %18 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add28 = add nsw i32 %18, %17
// CHECK-NEXT:   store i32 %add28, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next29
// CHECK: expand.next29:
// CHECK-NEXT:   store i32 8, ptr %y30, align 4
// CHECK-NEXT:   %19 = load i32, ptr %x23, align 4
// CHECK-NEXT:   %cmp31 = icmp eq i32 %19, 6
// CHECK-NEXT:   br i1 %cmp31, label %if.then32, label %if.end33
// CHECK: if.then32:
// CHECK-NEXT:   br label %expand.end35
// CHECK: if.end33:
// CHECK-NEXT:   %20 = load i32, ptr %y30, align 4
// CHECK-NEXT:   %21 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add34 = add nsw i32 %21, %20
// CHECK-NEXT:   store i32 %add34, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end35
// CHECK: expand.end35:
// CHECK-NEXT:   %22 = load i32, ptr %x23, align 4
// CHECK-NEXT:   %23 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add36 = add nsw i32 %23, %22
// CHECK-NEXT:   store i32 %add36, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next37
// CHECK: expand.next37:
// CHECK-NEXT:   store i32 6, ptr %x38, align 4
// CHECK-NEXT:   store i32 7, ptr %y39, align 4
// CHECK-NEXT:   %24 = load i32, ptr %x38, align 4
// CHECK-NEXT:   %cmp40 = icmp eq i32 %24, 6
// CHECK-NEXT:   br i1 %cmp40, label %if.then41, label %if.end42
// CHECK: if.then41:
// CHECK-NEXT:   br label %expand.next44
// CHECK: if.end42:
// CHECK-NEXT:   %25 = load i32, ptr %y39, align 4
// CHECK-NEXT:   %26 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add43 = add nsw i32 %26, %25
// CHECK-NEXT:   store i32 %add43, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next44
// CHECK: expand.next44:
// CHECK-NEXT:   store i32 8, ptr %y45, align 4
// CHECK-NEXT:   %27 = load i32, ptr %x38, align 4
// CHECK-NEXT:   %cmp46 = icmp eq i32 %27, 6
// CHECK-NEXT:   br i1 %cmp46, label %if.then47, label %if.end48
// CHECK: if.then47:
// CHECK-NEXT:   br label %expand.end50
// CHECK: if.end48:
// CHECK-NEXT:   %28 = load i32, ptr %y45, align 4
// CHECK-NEXT:   %29 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add49 = add nsw i32 %29, %28
// CHECK-NEXT:   store i32 %add49, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end50
// CHECK: expand.end50:
// CHECK-NEXT:   %30 = load i32, ptr %x38, align 4
// CHECK-NEXT:   %31 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add51 = add nsw i32 %31, %30
// CHECK-NEXT:   store i32 %add51, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end52
// CHECK: expand.end52:
// CHECK-NEXT:   %32 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %32


// CHECK-LABEL: define {{.*}} void @_Z5labelv()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x7 = alloca i32, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   %cmp = icmp eq i32 %0, 1
// CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
// CHECK: if.then:
// CHECK-NEXT:   br label %a
// CHECK: if.end:
// CHECK-NEXT:   %1 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 1, i32 {{.*}} %1)
// CHECK-NEXT:   br label %a
// CHECK: a:
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x1, align 4
// CHECK-NEXT:   %cmp2 = icmp eq i32 %2, 1
// CHECK-NEXT:   br i1 %cmp2, label %if.then3, label %if.end4
// CHECK: if.then3:
// CHECK-NEXT:   br label %a5
// CHECK: if.end4:
// CHECK-NEXT:   %3 = load i32, ptr %x1, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 1, i32 {{.*}} %3)
// CHECK-NEXT:   br label %a5
// CHECK: a5:
// CHECK-NEXT:   br label %expand.next6
// CHECK: expand.next6:
// CHECK-NEXT:   store i32 3, ptr %x7, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x7, align 4
// CHECK-NEXT:   %cmp8 = icmp eq i32 %4, 1
// CHECK-NEXT:   br i1 %cmp8, label %if.then9, label %if.end10
// CHECK: if.then9:
// CHECK-NEXT:   br label %a11
// CHECK: if.end10:
// CHECK-NEXT:   %5 = load i32, ptr %x7, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} 1, i32 {{.*}} %5)
// CHECK-NEXT:   br label %a11
// CHECK: a11:
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void
