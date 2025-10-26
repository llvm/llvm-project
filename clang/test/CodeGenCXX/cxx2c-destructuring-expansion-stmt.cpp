// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct A {};
struct B { int x = 1; };
struct C { int a = 1, b = 2, c = 3; };

void g(int);

int references_destructuring() {
  C c;
  template for (auto& x : c) { ++x; }
  template for (auto&& x : c) { ++x; }
  return c.a + c.b + c.c;
}

template <auto v>
int destructure() {
  int sum = 0;
  template for (auto x : v) sum += x;
  template for (constexpr auto x : v) sum += x;
  return sum;
}

void f() {
  destructure<B{10}>();
  destructure<C{}>();
  destructure<C{3, 4, 5}>();
}

void empty() {
  static constexpr A a;
  template for (auto x : A()) g(x);
  template for (auto& x : a) g(x);
  template for (auto&& x : A()) g(x);
  template for (constexpr auto x : a) g(x);
}

// CHECK: @_ZZ5emptyvE1a = internal constant %struct.A zeroinitializer, align 1
// CHECK: @_ZTAXtl1BLi10EEE = {{.*}} constant %struct.B { i32 10 }, comdat
// CHECK: @_ZTAXtl1CLi1ELi2ELi3EEE = {{.*}} constant %struct.C { i32 1, i32 2, i32 3 }, comdat
// CHECK: @_ZTAXtl1CLi3ELi4ELi5EEE = {{.*}} constant %struct.C { i32 3, i32 4, i32 5 }, comdat


// CHECK-LABEL: define {{.*}} i32 @_Z24references_destructuringv()
// CHECK: entry:
// CHECK-NEXT:   %c = alloca %struct.C, align 4
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca ptr, align 8
// CHECK-NEXT:   %x1 = alloca ptr, align 8
// CHECK-NEXT:   %x4 = alloca ptr, align 8
// CHECK-NEXT:   %1 = alloca ptr, align 8
// CHECK-NEXT:   %x7 = alloca ptr, align 8
// CHECK-NEXT:   %x11 = alloca ptr, align 8
// CHECK-NEXT:   %x15 = alloca ptr, align 8
// CHECK-NEXT:   call void @_ZN1CC1Ev(ptr {{.*}} %c)
// CHECK-NEXT:   store ptr %c, ptr %0, align 8
// CHECK-NEXT:   %2 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %a = getelementptr inbounds nuw %struct.C, ptr %2, i32 0, i32 0
// CHECK-NEXT:   store ptr %a, ptr %x, align 8
// CHECK-NEXT:   %3 = load ptr, ptr %x, align 8
// CHECK-NEXT:   %4 = load i32, ptr %3, align 4
// CHECK-NEXT:   %inc = add nsw i32 %4, 1
// CHECK-NEXT:   store i32 %inc, ptr %3, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %5 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %b = getelementptr inbounds nuw %struct.C, ptr %5, i32 0, i32 1
// CHECK-NEXT:   store ptr %b, ptr %x1, align 8
// CHECK-NEXT:   %6 = load ptr, ptr %x1, align 8
// CHECK-NEXT:   %7 = load i32, ptr %6, align 4
// CHECK-NEXT:   %inc2 = add nsw i32 %7, 1
// CHECK-NEXT:   store i32 %inc2, ptr %6, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   %8 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %c5 = getelementptr inbounds nuw %struct.C, ptr %8, i32 0, i32 2
// CHECK-NEXT:   store ptr %c5, ptr %x4, align 8
// CHECK-NEXT:   %9 = load ptr, ptr %x4, align 8
// CHECK-NEXT:   %10 = load i32, ptr %9, align 4
// CHECK-NEXT:   %inc6 = add nsw i32 %10, 1
// CHECK-NEXT:   store i32 %inc6, ptr %9, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr %c, ptr %1, align 8
// CHECK-NEXT:   %11 = load ptr, ptr %1, align 8
// CHECK-NEXT:   %a8 = getelementptr inbounds nuw %struct.C, ptr %11, i32 0, i32 0
// CHECK-NEXT:   store ptr %a8, ptr %x7, align 8
// CHECK-NEXT:   %12 = load ptr, ptr %x7, align 8
// CHECK-NEXT:   %13 = load i32, ptr %12, align 4
// CHECK-NEXT:   %inc9 = add nsw i32 %13, 1
// CHECK-NEXT:   store i32 %inc9, ptr %12, align 4
// CHECK-NEXT:   br label %expand.next10
// CHECK: expand.next10:
// CHECK-NEXT:   %14 = load ptr, ptr %1, align 8
// CHECK-NEXT:   %b12 = getelementptr inbounds nuw %struct.C, ptr %14, i32 0, i32 1
// CHECK-NEXT:   store ptr %b12, ptr %x11, align 8
// CHECK-NEXT:   %15 = load ptr, ptr %x11, align 8
// CHECK-NEXT:   %16 = load i32, ptr %15, align 4
// CHECK-NEXT:   %inc13 = add nsw i32 %16, 1
// CHECK-NEXT:   store i32 %inc13, ptr %15, align 4
// CHECK-NEXT:   br label %expand.next14
// CHECK: expand.next14:
// CHECK-NEXT:   %17 = load ptr, ptr %1, align 8
// CHECK-NEXT:   %c16 = getelementptr inbounds nuw %struct.C, ptr %17, i32 0, i32 2
// CHECK-NEXT:   store ptr %c16, ptr %x15, align 8
// CHECK-NEXT:   %18 = load ptr, ptr %x15, align 8
// CHECK-NEXT:   %19 = load i32, ptr %18, align 4
// CHECK-NEXT:   %inc17 = add nsw i32 %19, 1
// CHECK-NEXT:   store i32 %inc17, ptr %18, align 4
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   %a19 = getelementptr inbounds nuw %struct.C, ptr %c, i32 0, i32 0
// CHECK-NEXT:   %20 = load i32, ptr %a19, align 4
// CHECK-NEXT:   %b20 = getelementptr inbounds nuw %struct.C, ptr %c, i32 0, i32 1
// CHECK-NEXT:   %21 = load i32, ptr %b20, align 4
// CHECK-NEXT:   %add = add nsw i32 %20, %21
// CHECK-NEXT:   %c21 = getelementptr inbounds nuw %struct.C, ptr %c, i32 0, i32 2
// CHECK-NEXT:   %22 = load i32, ptr %c21, align 4
// CHECK-NEXT:   %add22 = add nsw i32 %add, %22
// CHECK-NEXT:   ret i32 %add22


// CHECK-LABEL: define {{.*}} void @_Z1fv()
// CHECK: entry:
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z11destructureITnDaXtl1BLi10EEEEiv()
// CHECK-NEXT:   %call1 = call {{.*}} i32 @_Z11destructureITnDaXtl1CLi1ELi2ELi3EEEEiv()
// CHECK-NEXT:   %call2 = call {{.*}} i32 @_Z11destructureITnDaXtl1CLi3ELi4ELi5EEEEiv()
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} i32 @_Z11destructureITnDaXtl1BLi10EEEEiv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %1 = alloca ptr, align 8
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZTAXtl1BLi10EEE, ptr %0, align 8
// CHECK-NEXT:   store i32 10, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr @_ZTAXtl1BLi10EEE, ptr %1, align 8
// CHECK-NEXT:   store i32 10, ptr %x1, align 4
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %4, 10
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end3
// CHECK: expand.end3:
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %5


// CHECK-LABEL: define {{.*}} i32 @_Z11destructureITnDaXtl1CLi1ELi2ELi3EEEEiv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %1 = alloca ptr, align 8
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZTAXtl1CLi1ELi2ELi3EEE, ptr %0, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 3, ptr %x4, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x4, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr @_ZTAXtl1CLi1ELi2ELi3EEE, ptr %1, align 8
// CHECK-NEXT:   store i32 1, ptr %x6, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %8, 1
// CHECK-NEXT:   store i32 %add7, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i32 2, ptr %x9, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add10 = add nsw i32 %9, 2
// CHECK-NEXT:   store i32 %add10, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i32 3, ptr %x12, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %10, 3
// CHECK-NEXT:   store i32 %add13, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %11


// CHECK-LABEL: define {{.*}} i32 @_Z11destructureITnDaXtl1CLi3ELi4ELi5EEEEiv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %1 = alloca ptr, align 8
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZTAXtl1CLi3ELi4ELi5EEE, ptr %0, align 8
// CHECK-NEXT:   store i32 3, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 4, ptr %x1, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 5, ptr %x4, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x4, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr @_ZTAXtl1CLi3ELi4ELi5EEE, ptr %1, align 8
// CHECK-NEXT:   store i32 3, ptr %x6, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %8, 3
// CHECK-NEXT:   store i32 %add7, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i32 4, ptr %x9, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add10 = add nsw i32 %9, 4
// CHECK-NEXT:   store i32 %add10, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i32 5, ptr %x12, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %10, 5
// CHECK-NEXT:   store i32 %add13, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %11


// CHECK-LABEL: define {{.*}} void @_Z5emptyv()
// CHECK: entry:
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %ref.tmp = alloca %struct.A, align 1
// CHECK-NEXT:   %1 = alloca ptr, align 8
// CHECK-NEXT:   %2 = alloca ptr, align 8
// CHECK-NEXT:   %ref.tmp1 = alloca %struct.A, align 1
// CHECK-NEXT:   %3 = alloca ptr, align 8
// CHECK-NEXT:   store ptr %ref.tmp, ptr %0, align 8
// CHECK-NEXT:   store ptr @_ZZ5emptyvE1a, ptr %1, align 8
// CHECK-NEXT:   store ptr %ref.tmp1, ptr %2, align 8
// CHECK-NEXT:   store ptr @_ZZ5emptyvE1a, ptr %3, align 8
// CHECK-NEXT:   ret void
