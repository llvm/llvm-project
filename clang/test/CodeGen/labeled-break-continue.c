// RUN: %clang_cc1 -std=c2y -triple x86_64-unknown-linux -emit-llvm -o - %s | FileCheck %s

bool g1();
bool g2();
bool g3();

// CHECK-LABEL: define {{.*}} void @f1()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.body
// CHECK: while.body:
// CHECK:   br label %while.end
// CHECK: while.end:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   br label %while.body1
// CHECK: while.body1:
// CHECK:   br label %while.body1
void f1() {
  l1: while (true) break l1;
  l2: while (true) continue l2;
}

// CHECK-LABEL: define {{.*}} void @f2()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   br label %for.end
// CHECK: for.end:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   br label %for.cond1
// CHECK: for.cond1:
// CHECK:   br label %for.cond1
void f2() {
  l1: for (;;) break l1;
  l2: for (;;) continue l2;
}

// CHECK-LABEL: define {{.*}} void @f3()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %do.body
// CHECK: do.body:
// CHECK:   br label %do.end
// CHECK: do.cond:
// CHECK:   br i1 true, label %do.body, label %do.end
// CHECK: do.end:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   br label %do.body1
// CHECK: do.body1:
// CHECK:   br label %do.cond2
// CHECK: do.cond2:
// CHECK:   br i1 true, label %do.body1, label %do.end3
// CHECK: do.end3:
// CHECK:   ret void
void f3() {
  l1: do { break l1; } while (true);
  l2: do { continue l2; } while (true);
}

// CHECK-LABEL: define {{.*}} void @f4()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @g1()
// CHECK:   br i1 %call, label %while.body, label %while.end14
// CHECK: while.body:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   br label %while.cond1
// CHECK: while.cond1:
// CHECK:   %call2 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call2, label %while.body3, label %while.end
// CHECK: while.body3:
// CHECK:   %call4 = call {{.*}} i1 @g3()
// CHECK:   br i1 %call4, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   br label %while.end14
// CHECK: if.end:
// CHECK:   %call5 = call {{.*}} i1 @g3()
// CHECK:   br i1 %call5, label %if.then6, label %if.end7
// CHECK: if.then6:
// CHECK:   br label %while.end
// CHECK: if.end7:
// CHECK:   %call8 = call {{.*}} i1 @g3()
// CHECK:   br i1 %call8, label %if.then9, label %if.end10
// CHECK: if.then9:
// CHECK:   br label %while.cond
// CHECK: if.end10:
// CHECK:   %call11 = call {{.*}} i1 @g3()
// CHECK:   br i1 %call11, label %if.then12, label %if.end13
// CHECK: if.then12:
// CHECK:   br label %while.cond1
// CHECK: if.end13:
// CHECK:   br label %while.cond1
// CHECK: while.end:
// CHECK:   br label %while.cond
// CHECK: while.end14:
// CHECK:   ret void
void f4() {
  l1: while (g1()) {
    l2: while (g2()) {
      if (g3()) break l1;
      if (g3()) break l2;
      if (g3()) continue l1;
      if (g3()) continue l2;
    }
  }
}

// CHECK-LABEL: define {{.*}} void @f5()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @g1()
// CHECK:   br i1 %call, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   %call1 = call {{.*}} i1 @g2()
// CHECK:   %conv = zext i1 %call1 to i32
// CHECK:   switch i32 %conv, label %sw.epilog [
// CHECK:     i32 1, label %sw.bb
// CHECK:     i32 2, label %sw.bb2
// CHECK:     i32 3, label %sw.bb3
// CHECK:   ]
// CHECK: sw.bb:
// CHECK:   br label %while.end
// CHECK: sw.bb2:
// CHECK:   br label %sw.epilog
// CHECK: sw.bb3:
// CHECK:   br label %while.cond
// CHECK: sw.epilog:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
void f5() {
  l1: while (g1()) {
    l2: switch (g2()) {
      case 1: break l1;
      case 2: break l2;
      case 3: continue l1;
    }
  }
}

// CHECK-LABEL: define {{.*}} void @f6()
// CHECK: entry:
// CHECK:   br label %l1
// CHECK: l1:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @g1()
// CHECK:   br i1 %call, label %while.body, label %while.end28
// CHECK: while.body:
// CHECK:   br label %l2
// CHECK: l2:
// CHECK:   br label %for.cond
// CHECK: for.cond:
// CHECK:   %call1 = call {{.*}} i1 @g1()
// CHECK:   br i1 %call1, label %for.body, label %for.end
// CHECK: for.body:
// CHECK:   br label %l3
// CHECK: l3:
// CHECK:   br label %do.body
// CHECK: do.body:
// CHECK:   br label %l4
// CHECK: l4:
// CHECK:   br label %while.cond2
// CHECK: while.cond2:
// CHECK:   %call3 = call {{.*}} i1 @g1()
// CHECK:   br i1 %call3, label %while.body4, label %while.end
// CHECK: while.body4:
// CHECK:   %call5 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call5, label %if.then, label %if.end
// CHECK: if.then:
// CHECK:   br label %while.end28
// CHECK: if.end:
// CHECK:   %call6 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call6, label %if.then7, label %if.end8
// CHECK: if.then7:
// CHECK:   br label %for.end
// CHECK: if.end8:
// CHECK:   %call9 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call9, label %if.then10, label %if.end11
// CHECK: if.then10:
// CHECK:   br label %do.end
// CHECK: if.end11:
// CHECK:   %call12 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call12, label %if.then13, label %if.end14
// CHECK: if.then13:
// CHECK:   br label %while.end
// CHECK: if.end14:
// CHECK:   %call15 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call15, label %if.then16, label %if.end17
// CHECK: if.then16:
// CHECK:   br label %while.cond
// CHECK: if.end17:
// CHECK:   %call18 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call18, label %if.then19, label %if.end20
// CHECK: if.then19:
// CHECK:   br label %for.cond
// CHECK: if.end20:
// CHECK:   %call21 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call21, label %if.then22, label %if.end23
// CHECK: if.then22:
// CHECK:   br label %do.cond
// CHECK: if.end23:
// CHECK:   %call24 = call {{.*}} i1 @g2()
// CHECK:   br i1 %call24, label %if.then25, label %if.end26
// CHECK: if.then25:
// CHECK:   br label %while.cond2
// CHECK: if.end26:
// CHECK:   br label %while.cond2
// CHECK: while.end:
// CHECK:   br label %do.cond
// CHECK: do.cond:
// CHECK:   %call27 = call {{.*}} i1 @g1()
// CHECK:   br i1 %call27, label %do.body, label %do.end
// CHECK: do.end:
// CHECK:   br label %for.cond
// CHECK: for.end:
// CHECK:   br label %while.cond
// CHECK: while.end28:
// CHECK:   ret void
void f6() {
  l1: while (g1()) {
    l2: for (; g1();) {
      l3: do {
        l4: while (g1()) {
          if (g2()) break l1;
          if (g2()) break l2;
          if (g2()) break l3;
          if (g2()) break l4;
          if (g2()) continue l1;
          if (g2()) continue l2;
          if (g2()) continue l3;
          if (g2()) continue l4;
        }
      } while (g1());
    }
  }
}

// CHECK-LABEL: define {{.*}} void @f7()
// CHECK: entry:
// CHECK:   br label %loop
// CHECK: loop:
// CHECK:   br label %while.cond
// CHECK: while.cond:
// CHECK:   %call = call {{.*}} i1 @g1()
// CHECK:   br i1 %call, label %while.body, label %while.end
// CHECK: while.body:
// CHECK:   %call1 = call {{.*}} i1 @g2()
// CHECK:   %conv = zext i1 %call1 to i32
// CHECK:   switch i32 %conv, label %sw.epilog [
// CHECK:     i32 1, label %sw.bb
// CHECK:   ]
// CHECK: sw.bb:
// CHECK:   br label %while.end
// CHECK: sw.epilog:
// CHECK:   br label %while.cond
// CHECK: while.end:
// CHECK:   ret void
void f7() {
  loop: while (g1()) {
    switch (g2()) {
      case 1: break loop;
    }
  }
}
