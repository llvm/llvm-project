// RUN: %clang_cc1 -triple x86_64-unknown-linux -std=c23 -fdefer-ts -emit-llvm %s -o - | FileCheck %s

void a();
void b();
void c();
void x(int q);

// CHECK-LABEL: define {{.*}} void @f1()
void f1() {
  // CHECK: call void @c()
  // CHECK: call void @b()
  // CHECK: call void @a()
  defer a();
  defer b();
  defer c();
}

// CHECK-LABEL: define {{.*}} void @f2()
void f2() {
  // CHECK: call void @x(i32 {{.*}} 1)
  // CHECK: call void @x(i32 {{.*}} 2)
  // CHECK: call void @x(i32 {{.*}} 3)
  // CHECK: call void @x(i32 {{.*}} 4)
  // CHECK: call void @x(i32 {{.*}} 5)
  defer x(5);
  {
    defer x(4);
    {
      defer x(2);
      defer x(1);
    }
    x(3);
  }
}

// CHECK-LABEL: define {{.*}} void @f3(i1 {{.*}} %ret)
void f3(bool ret) {
  // CHECK:   %ret.addr = alloca i8, align 1
  // CHECK:   %cleanup.dest.slot = alloca i32, align 4
  // CHECK:   %storedv = zext i1 %ret to i8
  // CHECK:   store i8 %storedv, ptr %ret.addr, align 1
  // CHECK:   %0 = load i8, ptr %ret.addr, align 1
  // CHECK:   %loadedv = trunc i8 %0 to i1
  // CHECK:   br i1 %loadedv, label %if.then, label %if.end
  // CHECK: if.then:
  // CHECK:   store i32 1, ptr %cleanup.dest.slot, align 4
  // CHECK:   br label %cleanup
  // CHECK: if.end:
  // CHECK:   call void @x(i32 noundef 1)
  // CHECK:   store i32 0, ptr %cleanup.dest.slot, align 4
  // CHECK:   br label %cleanup
  // CHECK: cleanup:
  // CHECK:   call void @x(i32 noundef 2)
  // CHECK:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
  // CHECK:   switch i32 %cleanup.dest, label %unreachable [
  // CHECK:     i32 0, label %cleanup.cont
  // CHECK:     i32 1, label %cleanup.cont
  // CHECK:   ]
  // CHECK: cleanup.cont:
  // CHECK:   ret void
  // CHECK: unreachable:
  // CHECK:   unreachable
  defer x(2);
  if (ret) return;
  defer x(1);
}
