// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZZ2f1vE1y = internal global i32 1, align 4
// CHECK: @_ZZ2f1vE1y_0 = internal global i32 2, align 4
// CHECK: @_ZZ2f1vE1y_1 = internal global i32 3, align 4
// CHECK: @_ZZ2f1vE1y_2 = internal global i32 4, align 4

// CHECK-LABEL: define {{.*}} i32 @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x7 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr @_ZZ2f1vE1y, align 4
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %1, %0
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %2 = load i32, ptr @_ZZ2f1vE1y_0, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 3, ptr %x4, align 4
// CHECK-NEXT:   %4 = load i32, ptr @_ZZ2f1vE1y_1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next6
// CHECK: expand.next6:
// CHECK-NEXT:   store i32 4, ptr %x7, align 4
// CHECK-NEXT:   %6 = load i32, ptr @_ZZ2f1vE1y_2, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add8, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %8
int f1() {
  int sum = 0;
  template for (constexpr auto x : {1, 2, 3, 4}) {
    static int y = x;
    sum += y;
  }
  return sum;
}

// CHECK-LABEL: define {{.*}} i32 @_Z2f2v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %ref.tmp = alloca %class.anon, align 1
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %ref.tmp2 = alloca %class.anon.0, align 1
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %ref.tmp7 = alloca %class.anon.2, align 1
// CHECK-NEXT:   %x11 = alloca i32, align 4
// CHECK-NEXT:   %ref.tmp12 = alloca %class.anon.4, align 1
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %call = call {{.*}} i32 @_ZZ2f2vENKUlvE_clEv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %0, %call
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %call3 = call {{.*}} i32 @_ZZ2f2vENKUlvE0_clEv(ptr {{.*}} %ref.tmp2)
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %1, %call3
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   store i32 3, ptr %x6, align 4
// CHECK-NEXT:   %call8 = call {{.*}} i32 @_ZZ2f2vENKUlvE1_clEv(ptr {{.*}} %ref.tmp7)
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add9 = add nsw i32 %2, %call8
// CHECK-NEXT:   store i32 %add9, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next10
// CHECK: expand.next10:
// CHECK-NEXT:   store i32 4, ptr %x11, align 4
// CHECK-NEXT:   %call13 = call {{.*}} i32 @_ZZ2f2vENKUlvE2_clEv(ptr {{.*}} %ref.tmp12)
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add14 = add nsw i32 %3, %call13
// CHECK-NEXT:   store i32 %add14, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %4
int f2() {
  int sum = 0;
  template for (constexpr auto x : {1, 2, 3, 4}) {
    sum += []{ return x; }();
  }
  return sum;
}

// CHECK-LABEL: define {{.*}} i32 @_ZZ2f2vENKUlvE_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   ret i32 1


// CHECK-LABEL: define {{.*}} i32 @_ZZ2f2vENKUlvE0_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   ret i32 2


// CHECK-LABEL: define {{.*}} i32 @_ZZ2f2vENKUlvE1_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   ret i32 3


// CHECK-LABEL: define {{.*}} i32 @_ZZ2f2vENKUlvE2_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   ret i32 4
