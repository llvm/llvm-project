// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct E {
  int x, y;
  constexpr E(int x, int y) : x{x}, y{y} {}
};

template <typename ...Es>
int unexpanded_pack_good(Es ...es) {
  int sum = 0;
  ([&] {
    template for (auto x : es) sum += x;
    template for (Es e : {{5, 6}, {7, 8}}) sum += e.x + e.y;
  }(), ...);
  return sum;
}

int unexpanded_pack() {
  return unexpanded_pack_good(E{1, 2}, E{3, 4});
}


// CHECK: %struct.E = type { i32, i32 }
// CHECK: %class.anon = type { ptr, ptr }
// CHECK: %class.anon.0 = type { ptr, ptr }


// CHECK-LABEL: define {{.*}} i32 @_Z15unexpanded_packv()
// CHECK: entry:
// CHECK-NEXT:   %agg.tmp = alloca %struct.E, align 4
// CHECK-NEXT:   %agg.tmp1 = alloca %struct.E, align 4
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %agg.tmp, i32 {{.*}} 1, i32 {{.*}} 2)
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %agg.tmp1, i32 {{.*}} 3, i32 {{.*}} 4)
// CHECK-NEXT:   %0 = load i64, ptr %agg.tmp, align 4
// CHECK-NEXT:   %1 = load i64, ptr %agg.tmp1, align 4
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z20unexpanded_pack_goodIJ1ES0_EEiDpT_(i64 %0, i64 %1)
// CHECK-NEXT:   ret i32 %call


// CHECK-LABEL: define {{.*}} i32 @_Z20unexpanded_pack_goodIJ1ES0_EEiDpT_(i64 %es.coerce, i64 %es.coerce2)
// CHECK: entry:
// CHECK-NEXT:   %es = alloca %struct.E, align 4
// CHECK-NEXT:   %es3 = alloca %struct.E, align 4
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %ref.tmp = alloca %class.anon, align 8
// CHECK-NEXT:   %ref.tmp4 = alloca %class.anon.0, align 8
// CHECK-NEXT:   store i64 %es.coerce, ptr %es, align 4
// CHECK-NEXT:   store i64 %es.coerce2, ptr %es3, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   %0 = getelementptr inbounds nuw %class.anon, ptr %ref.tmp, i32 0, i32 0
// CHECK-NEXT:   store ptr %es, ptr %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds nuw %class.anon, ptr %ref.tmp, i32 0, i32 1
// CHECK-NEXT:   store ptr %sum, ptr %1, align 8
// CHECK-NEXT:   call void @_ZZ20unexpanded_pack_goodIJ1ES0_EEiDpT_ENKUlvE0_clEv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   %2 = getelementptr inbounds nuw %class.anon.0, ptr %ref.tmp4, i32 0, i32 0
// CHECK-NEXT:   store ptr %es3, ptr %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds nuw %class.anon.0, ptr %ref.tmp4, i32 0, i32 1
// CHECK-NEXT:   store ptr %sum, ptr %3, align 8
// CHECK-NEXT:   call void @_ZZ20unexpanded_pack_goodIJ1ES0_EEiDpT_ENKUlvE_clEv(ptr {{.*}} %ref.tmp4)
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %4


// CHECK-LABEL: define {{.*}} void @_ZN1EC1Eii(ptr {{.*}} %this, i32 {{.*}} %x, i32 {{.*}} %y) {{.*}}
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x.addr = alloca i32, align 4
// CHECK-NEXT:   %y.addr = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 %x, ptr %x.addr, align 4
// CHECK-NEXT:   store i32 %y, ptr %y.addr, align 4
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   %0 = load i32, ptr %x.addr, align 4
// CHECK-NEXT:   %1 = load i32, ptr %y.addr, align 4
// CHECK-NEXT:   call void @_ZN1EC2Eii(ptr {{.*}} %this1, i32 {{.*}} %0, i32 {{.*}} %1)
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZZ20unexpanded_pack_goodIJ1ES0_EEiDpT_ENKUlvE0_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %e = alloca %struct.E, align 4
// CHECK-NEXT:   %e10 = alloca %struct.E, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds nuw %class.anon, ptr %this1, i32 0, i32 0
// CHECK-NEXT:   %2 = load ptr, ptr %1, align 8
// CHECK-NEXT:   store ptr %2, ptr %0, align 8
// CHECK-NEXT:   %3 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %x2 = getelementptr inbounds nuw %struct.E, ptr %3, i32 0, i32 0
// CHECK-NEXT:   %4 = load i32, ptr %x2, align 4
// CHECK-NEXT:   store i32 %4, ptr %x, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds nuw %class.anon, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %7 = load ptr, ptr %6, align 8
// CHECK-NEXT:   %8 = load i32, ptr %7, align 4
// CHECK-NEXT:   %add = add nsw i32 %8, %5
// CHECK-NEXT:   store i32 %add, ptr %7, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %9 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %y = getelementptr inbounds nuw %struct.E, ptr %9, i32 0, i32 1
// CHECK-NEXT:   %10 = load i32, ptr %y, align 4
// CHECK-NEXT:   store i32 %10, ptr %x3, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds nuw %class.anon, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %13 = load ptr, ptr %12, align 8
// CHECK-NEXT:   %14 = load i32, ptr %13, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %14, %11
// CHECK-NEXT:   store i32 %add4, ptr %13, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %e, i32 {{.*}} 5, i32 {{.*}} 6)
// CHECK-NEXT:   %x5 = getelementptr inbounds nuw %struct.E, ptr %e, i32 0, i32 0
// CHECK-NEXT:   %15 = load i32, ptr %x5, align 4
// CHECK-NEXT:   %y6 = getelementptr inbounds nuw %struct.E, ptr %e, i32 0, i32 1
// CHECK-NEXT:   %16 = load i32, ptr %y6, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %15, %16
// CHECK-NEXT:   %17 = getelementptr inbounds nuw %class.anon, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %18 = load ptr, ptr %17, align 8
// CHECK-NEXT:   %19 = load i32, ptr %18, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %19, %add7
// CHECK-NEXT:   store i32 %add8, ptr %18, align 4
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %e10, i32 {{.*}} 7, i32 {{.*}} 8)
// CHECK-NEXT:   %x11 = getelementptr inbounds nuw %struct.E, ptr %e10, i32 0, i32 0
// CHECK-NEXT:   %20 = load i32, ptr %x11, align 4
// CHECK-NEXT:   %y12 = getelementptr inbounds nuw %struct.E, ptr %e10, i32 0, i32 1
// CHECK-NEXT:   %21 = load i32, ptr %y12, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %20, %21
// CHECK-NEXT:   %22 = getelementptr inbounds nuw %class.anon, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %23 = load ptr, ptr %22, align 8
// CHECK-NEXT:   %24 = load i32, ptr %23, align 4
// CHECK-NEXT:   %add14 = add nsw i32 %24, %add13
// CHECK-NEXT:   store i32 %add14, ptr %23, align 4
// CHECK-NEXT:   br label %expand.end15
// CHECK: expand.end15:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZZ20unexpanded_pack_goodIJ1ES0_EEiDpT_ENKUlvE_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %0 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %e = alloca %struct.E, align 4
// CHECK-NEXT:   %e10 = alloca %struct.E, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds nuw %class.anon.0, ptr %this1, i32 0, i32 0
// CHECK-NEXT:   %2 = load ptr, ptr %1, align 8
// CHECK-NEXT:   store ptr %2, ptr %0, align 8
// CHECK-NEXT:   %3 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %x2 = getelementptr inbounds nuw %struct.E, ptr %3, i32 0, i32 0
// CHECK-NEXT:   %4 = load i32, ptr %x2, align 4
// CHECK-NEXT:   store i32 %4, ptr %x, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds nuw %class.anon.0, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %7 = load ptr, ptr %6, align 8
// CHECK-NEXT:   %8 = load i32, ptr %7, align 4
// CHECK-NEXT:   %add = add nsw i32 %8, %5
// CHECK-NEXT:   store i32 %add, ptr %7, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %9 = load ptr, ptr %0, align 8
// CHECK-NEXT:   %y = getelementptr inbounds nuw %struct.E, ptr %9, i32 0, i32 1
// CHECK-NEXT:   %10 = load i32, ptr %y, align 4
// CHECK-NEXT:   store i32 %10, ptr %x3, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds nuw %class.anon.0, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %13 = load ptr, ptr %12, align 8
// CHECK-NEXT:   %14 = load i32, ptr %13, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %14, %11
// CHECK-NEXT:   store i32 %add4, ptr %13, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %e, i32 {{.*}} 5, i32 {{.*}} 6)
// CHECK-NEXT:   %x5 = getelementptr inbounds nuw %struct.E, ptr %e, i32 0, i32 0
// CHECK-NEXT:   %15 = load i32, ptr %x5, align 4
// CHECK-NEXT:   %y6 = getelementptr inbounds nuw %struct.E, ptr %e, i32 0, i32 1
// CHECK-NEXT:   %16 = load i32, ptr %y6, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %15, %16
// CHECK-NEXT:   %17 = getelementptr inbounds nuw %class.anon.0, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %18 = load ptr, ptr %17, align 8
// CHECK-NEXT:   %19 = load i32, ptr %18, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %19, %add7
// CHECK-NEXT:   store i32 %add8, ptr %18, align 4
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   call void @_ZN1EC1Eii(ptr {{.*}} %e10, i32 {{.*}} 7, i32 {{.*}} 8)
// CHECK-NEXT:   %x11 = getelementptr inbounds nuw %struct.E, ptr %e10, i32 0, i32 0
// CHECK-NEXT:   %20 = load i32, ptr %x11, align 4
// CHECK-NEXT:   %y12 = getelementptr inbounds nuw %struct.E, ptr %e10, i32 0, i32 1
// CHECK-NEXT:   %21 = load i32, ptr %y12, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %20, %21
// CHECK-NEXT:   %22 = getelementptr inbounds nuw %class.anon.0, ptr %this1, i32 0, i32 1
// CHECK-NEXT:   %23 = load ptr, ptr %22, align 8
// CHECK-NEXT:   %24 = load i32, ptr %23, align 4
// CHECK-NEXT:   %add14 = add nsw i32 %24, %add13
// CHECK-NEXT:   store i32 %add14, ptr %23, align 4
// CHECK-NEXT:   br label %expand.end15
// CHECK: expand.end15:
// CHECK-NEXT:   ret void
