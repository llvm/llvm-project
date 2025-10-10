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
  // CHECK:   call void @x(i32 {{.*}} 1)
  // CHECK:   store i32 0, ptr %cleanup.dest.slot, align 4
  // CHECK:   br label %cleanup
  // CHECK: cleanup:
  // CHECK:   call void @x(i32 {{.*}} 2)
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

// CHECK-LABEL: define {{.*}} void @ts_g()
void ts_g() {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
  return;
  defer x(42);
}

// CHECK-LABEL: define {{.*}} void @ts_h()
void ts_h() {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   br label %b
  // CHECK-EMPTY:
  goto b;
  {
    defer x(42);
  }

  // CHECK-NEXT: b:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
  b:
}

// CHECK-LABEL: define {{.*}} void @ts_i()
void ts_i() {
  // CHECK: entry:
  // CHECK:   %cleanup.dest.slot = alloca i32, align 4
  // CHECK:   store i32 2, ptr %cleanup.dest.slot, align 4
  // CHECK:   call void @x(i32 {{.*}} 42)
  // CHECK:   %cleanup.dest = load i32, ptr %cleanup.dest.slot, align 4
  // CHECK:   switch i32 %cleanup.dest, label %unreachable [
  // CHECK:     i32 2, label %b
  // CHECK:   ]
  // CHECK: b:
  // CHECK:   ret void
  // CHECK: unreachable:
  // CHECK:   unreachable
  {
    defer { x(42); }
    goto b;
  }
  b:
}


// CHECK-LABEL: define {{.*}} void @ts_m()
void ts_m() {
  // CHECK: entry:
  // CHECK:   br label %b
  // CHECK: b:
  // CHECK:   call void @x(i32 {{.*}} 1)
  // CHECK:   ret void
  goto b;
  {
    b:
    defer x(1);
  }
}

// CHECK-LABEL: define {{.*}} void @ts_p()
void ts_p() {
  // CHECK: entry:
  // CHECK:   br label %b
  // CHECK: b:
  // CHECK:   ret void
  {
    goto b;
    defer x(42);
  }
  b:
}

// CHECK-LABEL: define {{.*}} void @ts_r()
void ts_r() {
  // CHECK: entry:
  // CHECK:   br label %b
  // CHECK: b:
  // CHECK:   call void @x(i32 {{.*}} 42)
  // CHECK:   br label %b
  {
    b:
    defer x(42);
  }
  goto b;
}

// CHECK-LABEL: define {{.*}} i32 @return_value()
int return_value() {
  // CHECK: entry:
  // CHECK:   %r = alloca i32, align 4
  // CHECK:   %p = alloca ptr, align 8
  // CHECK:   store i32 4, ptr %r, align 4
  // CHECK:   store ptr %r, ptr %p, align 8
  // CHECK:   %0 = load ptr, ptr %p, align 8
  // CHECK:   %1 = load i32, ptr %0, align 4
  // CHECK:   %2 = load ptr, ptr %p, align 8
  // CHECK:   store i32 5, ptr %2, align 4
  // CHECK:   ret i32 %1
  int r = 4;
  int* p = &r;
  defer { *p = 5; }
  return *p;
}

void* malloc(__SIZE_TYPE__ size);
void free(void* ptr);
int use_buffer(__SIZE_TYPE__ size, void* ptr);

// CHECK-LABEL: define {{.*}} i32 @malloc_free_example()
int malloc_free_example() {
  // CHECK: entry:
  // CHECK:   %size = alloca i32, align 4
  // CHECK:   %buf = alloca ptr, align 8
  // CHECK:   store i32 20, ptr %size, align 4
  // CHECK:   %call = call ptr @malloc(i64 {{.*}} 20)
  // CHECK:   store ptr %call, ptr %buf, align 8
  // CHECK:   %0 = load ptr, ptr %buf, align 8
  // CHECK:   %call1 = call i32 @use_buffer(i64 {{.*}} 20, ptr {{.*}} %0)
  // CHECK:   %1 = load ptr, ptr %buf, align 8
  // CHECK:   call void @free(ptr {{.*}} %1)
  // CHECK:   ret i32 %call1
  const int size = 20;
  void* buf = malloc(size);
  defer { free(buf); }
  return use_buffer(size, buf);
}

// CHECK-LABEL: define {{.*}} void @sequencing_1()
void sequencing_1() {
  // CHECK: entry:
  // CHECK:   call void @x(i32 {{.*}} 1)
  // CHECK:   call void @x(i32 {{.*}} 2)
  // CHECK:   call void @x(i32 {{.*}} 3)
  // CHECK:   ret void
  {
    defer {
      x(3);
    }
    if (true)
      defer x(1);
    x(2);
  }
}

// CHECK-LABEL: define {{.*}} void @sequencing_2()
void sequencing_2() {
  // CHECK: entry:
  // CHECK:   %arr = alloca [3 x i32], align 4
  // CHECK:   %i = alloca i32, align 4
  // CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %arr, ptr align 4 @__const.sequencing_2.arr, i64 12, i1 false)
  // CHECK:   store i32 0, ptr %i, align 4
  // CHECK:   br label %for.cond
  // CHECK: for.cond:
  // CHECK:   %0 = load i32, ptr %i, align 4
  // CHECK:   %cmp = icmp ult i32 %0, 3
  // CHECK:   br i1 %cmp, label %for.body, label %for.end
  // CHECK: for.body:
  // CHECK:   %1 = load i32, ptr %i, align 4
  // CHECK:   %idxprom = zext i32 %1 to i64
  // CHECK:   %arrayidx = getelementptr inbounds nuw [3 x i32], ptr %arr, i64 0, i64 %idxprom
  // CHECK:   %2 = load i32, ptr %arrayidx, align 4
  // CHECK:   call void @x(i32 {{.*}} %2)
  // CHECK:   br label %for.inc
  // CHECK: for.inc:
  // CHECK:   %3 = load i32, ptr %i, align 4
  // CHECK:   %inc = add i32 %3, 1
  // CHECK:   store i32 %inc, ptr %i, align 4
  // CHECK:   br label %for.cond
  // CHECK: for.end:
  // CHECK:   call void @x(i32 {{.*}} 4)
  // CHECK:   call void @x(i32 {{.*}} 5)
  // CHECK:   ret void
  {
    int arr[] = {1, 2, 3};
    defer {
      x(5);
    }
    for (unsigned i = 0; i < 3; ++i)
      defer x(arr[i]);
    x(4);
  }
}

// CHECK-LABEL: define {{.*}} void @sequencing_3()
void sequencing_3() {
  // CHECK: entry:
  // CHECK:   %r = alloca i32, align 4
  // CHECK:   store i32 0, ptr %r, align 4
  // CHECK:   %0 = load i32, ptr %r, align 4
  // CHECK:   %add = add nsw i32 %0, 1
  // CHECK:   store i32 %add, ptr %r, align 4
  // CHECK:   %1 = load i32, ptr %r, align 4
  // CHECK:   %mul = mul nsw i32 %1, 2
  // CHECK:   store i32 %mul, ptr %r, align 4
  // CHECK:   %2 = load i32, ptr %r, align 4
  // CHECK:   %add1 = add nsw i32 %2, 3
  // CHECK:   store i32 %add1, ptr %r, align 4
  // CHECK:   %3 = load i32, ptr %r, align 4
  // CHECK:   %mul2 = mul nsw i32 %3, 4
  // CHECK:   store i32 %mul2, ptr %r, align 4
  // CHECK:   ret void
  int r = 0;
  {
    defer {
      defer r *= 4;
      r *= 2;
      defer {
        r += 3;
      }
    }
    defer r += 1;
  }
}

// CHECK-LABEL: define {{.*}} void @defer_stmt(i32 {{.*}} %q)
void defer_stmt(int q) {
  // CHECK: entry:
  // CHECK:   %q.addr = alloca i32, align 4
  // CHECK:   store i32 %q, ptr %q.addr, align 4
  // CHECK:   %0 = load i32, ptr %q.addr, align 4
  // CHECK:   %cmp = icmp eq i32 %0, 3
  // CHECK:   br i1 %cmp, label %if.then, label %if.end
  // CHECK: if.then:
  // CHECK:   call void @x(i32 {{.*}} 42)
  // CHECK:   br label %if.end
  // CHECK: if.end:
  // CHECK:   ret void
  defer if (q == 3) x(42);
}

// CHECK-LABEL: define {{.*}} void @defer_defer()
void defer_defer() {
  // CHECK: entry:
  // CHECK:   call void @x(i32 {{.*}} 0)
  // CHECK:   call void @x(i32 {{.*}} 1)
  // CHECK:   call void @x(i32 {{.*}} 2)
  // CHECK:   call void @x(i32 {{.*}} 3)
  // CHECK:   call void @x(i32 {{.*}} 4)
  // CHECK:   ret void
  defer x(4);
  defer defer x(3);
  defer defer defer x(2);
  defer defer defer defer x(1);
  x(0);
}

// CHECK-LABEL: define {{.*}} i32 @vla(ptr {{.*}} %p, i32 {{.*}} %x)
int vla(int* p, int x) {
    // CHECK: entry:
    // CHECK:   %retval = alloca i32, align 4
    // CHECK:   %p.addr = alloca ptr, align 8
    // CHECK:   %x.addr = alloca i32, align 4
    // CHECK:   %cleanup.dest.slot = alloca i32, align 4
    // CHECK:   %saved_stack = alloca ptr, align 8
    // CHECK:   %__vla_expr0 = alloca i64, align 8
    // CHECK:   %saved_stack2 = alloca ptr, align 8
    // CHECK:   %__vla_expr1 = alloca i64, align 8
    // CHECK:   store ptr %p, ptr %p.addr, align 8
    // CHECK:   store i32 %x, ptr %x.addr, align 4
    // CHECK:   %0 = load i32, ptr %x.addr, align 4
    // CHECK:   %cmp = icmp slt i32 %0, 5
    // CHECK:   br i1 %cmp, label %if.then, label %if.end
    // CHECK: if.then:
    // CHECK:   store i32 10, ptr %retval, align 4
    // CHECK:   store i32 1, ptr %cleanup.dest.slot, align 4
    // CHECK:   br label %cleanup
    // CHECK: if.end:
    // CHECK:   store i32 7, ptr %retval, align 4
    // CHECK:   store i32 1, ptr %cleanup.dest.slot, align 4
    // CHECK:   %1 = load i32, ptr %x.addr, align 4
    // CHECK:   %2 = zext i32 %1 to i64
    // CHECK:   %3 = call ptr @llvm.stacksave.p0()
    // CHECK:   store ptr %3, ptr %saved_stack, align 8
    // CHECK:   %vla = alloca i32, i64 %2, align 16
    // CHECK:   store i64 %2, ptr %__vla_expr0, align 8
    // CHECK:   %arrayidx = getelementptr inbounds i32, ptr %vla, i64 2
    // CHECK:   store i32 4, ptr %arrayidx, align 8
    // CHECK:   %arrayidx1 = getelementptr inbounds i32, ptr %vla, i64 2
    // CHECK:   %4 = load i32, ptr %arrayidx1, align 8
    // CHECK:   %5 = load ptr, ptr %p.addr, align 8
    // CHECK:   store i32 %4, ptr %5, align 4
    // CHECK:   %6 = load ptr, ptr %saved_stack, align 8
    // CHECK:   call void @llvm.stackrestore.p0(ptr %6)
    // CHECK:   br label %cleanup
    // CHECK: cleanup:
    // CHECK:   %7 = load i32, ptr %x.addr, align 4
    // CHECK:   %8 = zext i32 %7 to i64
    // CHECK:   %9 = call ptr @llvm.stacksave.p0()
    // CHECK:   store ptr %9, ptr %saved_stack2, align 8
    // CHECK:   %vla3 = alloca i32, i64 %8, align 16
    // CHECK:   store i64 %8, ptr %__vla_expr1, align 8
    // CHECK:   %arrayidx4 = getelementptr inbounds i32, ptr %vla3, i64 2
    // CHECK:   store i32 3, ptr %arrayidx4, align 8
    // CHECK:   %arrayidx5 = getelementptr inbounds i32, ptr %vla3, i64 2
    // CHECK:   %10 = load i32, ptr %arrayidx5, align 8
    // CHECK:   %11 = load ptr, ptr %p.addr, align 8
    // CHECK:   store i32 %10, ptr %11, align 4
    // CHECK:   %12 = load ptr, ptr %saved_stack2, align 8
    // CHECK:   call void @llvm.stackrestore.p0(ptr %12)
    // CHECK:   %13 = load i32, ptr %retval, align 4
    // CHECK:   ret i32 %13
	defer {
		int a[x];
		a[2] = 3;
		*p = a[2];
	}
	if (x < 5) { return 10; }
	defer {
		int b[x];
		b[2] = 4;
		*p = b[2];
	}
	return 7;
}

[[noreturn]] void exit();
[[noreturn]] void _Exit();
[[noreturn]] void foobar();

// CHECK-LABEL: define {{.*}} i32 @call_exit()
int call_exit() {
    // CHECK: entry:
    // CHECK:   call void @exit()
    // CHECK:   unreachable
    defer x(1);
    exit();
}

// CHECK-LABEL: define {{.*}} i32 @call__Exit()
int call__Exit() {
    // CHECK: entry:
    // CHECK:   call void @_Exit()
    // CHECK:   unreachable
    defer x(1);
    _Exit();
}

// CHECK-LABEL: define {{.*}} i32 @call_foobar()
int call_foobar() {
    // CHECK: entry:
    // CHECK:   call void @foobar()
    // CHECK:   unreachable
    defer x(1);
    foobar();
}

// CHECK-LABEL: define {{.*}} i32 @main()
int main() {
  // CHECK: entry:
  // CHECK:   %retval = alloca i32, align 4
  // CHECK:   store i32 0, ptr %retval, align 4
  // CHECK:   store i32 5, ptr %retval, align 4
  // CHECK:   call void @x(i32 {{.*}} 42)
  // CHECK:   %0 = load i32, ptr %retval, align 4
  // CHECK:   ret i32 %0
  defer x(42);
  return 5;
}
