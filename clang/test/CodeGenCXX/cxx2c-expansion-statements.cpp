// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S {
  int x;
  constexpr S(int x) : x{x} {}
};

void g(int);
void g(long);
void g(const char*);
void g(S);

template <int n> constexpr int tg() { return n; }

void h(int, int);

void f1() {
  template for (auto x : {1, 2, 3}) g(x);
}

void f2() {
  template for (auto x : {1, "123", S(45)}) g(x);
}

void f3() {
  template for (auto x : {}) g(x);
}

void f4() {
  template for (auto x : {1, 2})
    template for (auto y : {3, 4})
      h(x, y);
}

void f5() {
  template for (auto x : {}) static_assert(false, "discarded");
  template for (constexpr auto x : {}) static_assert(false, "discarded");
  template for (auto x : {1}) g(x);
  template for (auto x : {2, 3, 4}) g(x);
  template for (constexpr auto x : {5}) g(x);
  template for (constexpr auto x : {6, 7, 8}) g(x);
  template for (constexpr auto x : {9}) tg<x>();
  template for (constexpr auto x : {10, 11, 12})
    static_assert(tg<x>());

  template for (int x : {13, 14, 15}) g(x);
  template for (S x : {16, 17, 18}) g(x.x);
  template for (constexpr S x : {19, 20, 21}) tg<x.x>();
}

template <typename T>
void t1() {
  template for (T x : {}) g(x);
  template for (constexpr T x : {}) g(x);
  template for (auto x : {}) g(x);
  template for (constexpr auto x : {}) g(x);
  template for (T x : {1, 2}) g(x);
  template for (T x : {T(3), T(4)}) g(x);
  template for (auto x : {T(5), T(6)}) g(x);
  template for (constexpr T x : {T(7), T(8)}) static_assert(tg<x>());
  template for (constexpr auto x : {T(9), T(10)}) static_assert(tg<x>());
}

template <typename U>
struct s1 {
  template <typename T>
  void tf() {
    template for (T x : {}) g(x);
    template for (constexpr T x : {}) g(x);
    template for (U x : {}) g(x);
    template for (constexpr U x : {}) g(x);
    template for (auto x : {}) g(x);
    template for (constexpr auto x : {}) g(x);
    template for (T x : {1, 2}) g(x);
    template for (U x : {3, 4}) g(x);
    template for (U x : {T(5), T(6)}) g(x);
    template for (T x : {U(7), U(8)}) g(x);
    template for (auto x : {T(9), T(10)}) g(x);
    template for (auto x : {U(11), T(12)}) g(x);
    template for (constexpr U x : {T(13), T(14)}) static_assert(tg<x>());
    template for (constexpr T x : {U(15), U(16)}) static_assert(tg<x>());
    template for (constexpr auto x : {T(17), U(18)}) static_assert(tg<x>());
  }
};

template <typename T>
void t2() {
  template for (T x : {}) g(x);
}

void f6() {
  t1<int>();
  t1<long>();
  s1<long>().tf<long>();
  s1<int>().tf<int>();
  s1<int>().tf<long>();
  s1<long>().tf<int>();
  t2<S>();
  t2<S[1231]>();
  t2<S***>();
}

// CHECK-LABEL: define {{.*}} void @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x1, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.next2
// CHECK: expand.next2:
// CHECK-NEXT:   store i32 3, ptr %x3, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x3, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f2v()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca ptr, align 8
// CHECK-NEXT:   %x3 = alloca %struct.S, align 4
// CHECK-NEXT:   %agg.tmp = alloca %struct.S, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store ptr @.str, ptr %x1, align 8
// CHECK-NEXT:   %1 = load ptr, ptr %x1, align 8
// CHECK-NEXT:   call void @_Z1gPKc(ptr {{.*}} %1)
// CHECK-NEXT:   br label %expand.next2
// CHECK: expand.next2:
// CHECK-NEXT:   call void @_ZN1SC1Ei(ptr {{.*}} %x3, i32 {{.*}} 45)
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x3, i64 4, i1 false)
// CHECK-NEXT:   %coerce.dive = getelementptr inbounds nuw %struct.S, ptr %agg.tmp, i32 0, i32 0
// CHECK-NEXT:   %2 = load i32, ptr %coerce.dive, align 4
// CHECK-NEXT:   call void @_Z1g1S(i32 %2)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f3v()
// CHECK: entry:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f4v()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %y1 = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %y4 = alloca i32, align 4
// CHECK-NEXT:   %y6 = alloca i32, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   store i32 3, ptr %y, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %y, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} %0, i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 4, ptr %y1, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %y1, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} %2, i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   br label %expand.next2
// CHECK: expand.next2:
// CHECK-NEXT:   store i32 2, ptr %x3, align 4
// CHECK-NEXT:   store i32 3, ptr %y4, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %5 = load i32, ptr %y4, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} %4, i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   store i32 4, ptr %y6, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %7 = load i32, ptr %y6, align 4
// CHECK-NEXT:   call void @_Z1hii(i32 {{.*}} %6, i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end7
// CHECK: expand.end7:
// CHECK-NEXT:   br label %expand.end8
// CHECK: expand.end8:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f5v()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   %x10 = alloca i32, align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   %x14 = alloca i32, align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK-NEXT:   %x18 = alloca i32, align 4
// CHECK-NEXT:   %x20 = alloca i32, align 4
// CHECK-NEXT:   %x22 = alloca i32, align 4
// CHECK-NEXT:   %x24 = alloca i32, align 4
// CHECK-NEXT:   %x26 = alloca i32, align 4
// CHECK-NEXT:   %x28 = alloca %struct.S, align 4
// CHECK-NEXT:   %x31 = alloca %struct.S, align 4
// CHECK-NEXT:   %x34 = alloca %struct.S, align 4
// CHECK-NEXT:   %x37 = alloca %struct.S, align 4
// CHECK-NEXT:   %x40 = alloca %struct.S, align 4
// CHECK-NEXT:   %x43 = alloca %struct.S, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x1, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 3, ptr %x2, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x2, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 4, ptr %x4, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x4, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end5
// CHECK: expand.end5:
// CHECK-NEXT:   store i32 5, ptr %x6, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 5)
// CHECK-NEXT:   br label %expand.end7
// CHECK: expand.end7:
// CHECK-NEXT:   store i32 6, ptr %x8, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 6)
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   store i32 7, ptr %x10, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 7)
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i32 8, ptr %x12, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 8)
// CHECK-NEXT:   br label %expand.end13
// CHECK: expand.end13:
// CHECK-NEXT:   store i32 9, ptr %x14, align 4
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi9EEiv()
// CHECK-NEXT:   br label %expand.end15
// CHECK: expand.end15:
// CHECK-NEXT:   store i32 10, ptr %x16, align 4
// CHECK-NEXT:   br label %expand.next17
// CHECK: expand.next17:
// CHECK-NEXT:   store i32 11, ptr %x18, align 4
// CHECK-NEXT:   br label %expand.next19
// CHECK: expand.next19:
// CHECK-NEXT:   store i32 12, ptr %x20, align 4
// CHECK-NEXT:   br label %expand.end21
// CHECK: expand.end21:
// CHECK-NEXT:   store i32 13, ptr %x22, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x22, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next23
// CHECK: expand.next23:
// CHECK-NEXT:   store i32 14, ptr %x24, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x24, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.next25
// CHECK: expand.next25:
// CHECK-NEXT:   store i32 15, ptr %x26, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x26, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %6)
// CHECK-NEXT:   br label %expand.end27
// CHECK: expand.end27:
// CHECK-NEXT:   call void @_ZN1SC1Ei(ptr {{.*}} %x28, i32 {{.*}} 16)
// CHECK-NEXT:   %x29 = getelementptr inbounds nuw %struct.S, ptr %x28, i32 0, i32 0
// CHECK-NEXT:   %7 = load i32, ptr %x29, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.next30
// CHECK: expand.next30:
// CHECK-NEXT:   call void @_ZN1SC1Ei(ptr {{.*}} %x31, i32 {{.*}} 17)
// CHECK-NEXT:   %x32 = getelementptr inbounds nuw %struct.S, ptr %x31, i32 0, i32 0
// CHECK-NEXT:   %8 = load i32, ptr %x32, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next33
// CHECK: expand.next33:
// CHECK-NEXT:   call void @_ZN1SC1Ei(ptr {{.*}} %x34, i32 {{.*}} 18)
// CHECK-NEXT:   %x35 = getelementptr inbounds nuw %struct.S, ptr %x34, i32 0, i32 0
// CHECK-NEXT:   %9 = load i32, ptr %x35, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %9)
// CHECK-NEXT:   br label %expand.end36
// CHECK: expand.end36:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x37, ptr align 4 @__const._Z2f5v.x, i64 4, i1 false)
// CHECK-NEXT:   %call38 = call {{.*}} i32 @_Z2tgILi19EEiv()
// CHECK-NEXT:   br label %expand.next39
// CHECK: expand.next39:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x40, ptr align 4 @__const._Z2f5v.x.1, i64 4, i1 false)
// CHECK-NEXT:   %call41 = call {{.*}} i32 @_Z2tgILi20EEiv()
// CHECK-NEXT:   br label %expand.next42
// CHECK: expand.next42:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x43, ptr align 4 @__const._Z2f5v.x.2, i64 4, i1 false)
// CHECK-NEXT:   %call44 = call {{.*}} i32 @_Z2tgILi21EEiv()
// CHECK-NEXT:   br label %expand.end45
// CHECK: expand.end45:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f6v()
// CHECK: entry:
// CHECK-NEXT:   %ref.tmp = alloca %struct.s1, align 1
// CHECK-NEXT:   %ref.tmp1 = alloca %struct.s1.0, align 1
// CHECK-NEXT:   %ref.tmp2 = alloca %struct.s1.0, align 1
// CHECK-NEXT:   %ref.tmp3 = alloca %struct.s1, align 1
// CHECK-NEXT:   call void @_Z2t1IiEvv()
// CHECK-NEXT:   call void @_Z2t1IlEvv()
// CHECK-NEXT:   call void @_ZN2s1IlE2tfIlEEvv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   call void @_ZN2s1IiE2tfIiEEvv(ptr {{.*}} %ref.tmp1)
// CHECK-NEXT:   call void @_ZN2s1IiE2tfIlEEvv(ptr {{.*}} %ref.tmp2)
// CHECK-NEXT:   call void @_ZN2s1IlE2tfIiEEvv(ptr {{.*}} %ref.tmp3)
// CHECK-NEXT:   call void @_Z2t2I1SEvv()
// CHECK-NEXT:   call void @_Z2t2IA1231_1SEvv()
// CHECK-NEXT:   call void @_Z2t2IPPP1SEvv()
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2t1IiEvv()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   %x10 = alloca i32, align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   %x14 = alloca i32, align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x1, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 3, ptr %x2, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x2, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 4, ptr %x4, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x4, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end5
// CHECK: expand.end5:
// CHECK-NEXT:   store i32 5, ptr %x6, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x6, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next7
// CHECK: expand.next7:
// CHECK-NEXT:   store i32 6, ptr %x8, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x8, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end9
// CHECK: expand.end9:
// CHECK-NEXT:   store i32 7, ptr %x10, align 4
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i32 8, ptr %x12, align 4
// CHECK-NEXT:   br label %expand.end13
// CHECK: expand.end13:
// CHECK-NEXT:   store i32 9, ptr %x14, align 4
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   store i32 10, ptr %x16, align 4
// CHECK-NEXT:   br label %expand.end17
// CHECK: expand.end17:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2t1IlEvv()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i64, align 8
// CHECK-NEXT:   %x1 = alloca i64, align 8
// CHECK-NEXT:   %x2 = alloca i64, align 8
// CHECK-NEXT:   %x4 = alloca i64, align 8
// CHECK-NEXT:   %x6 = alloca i64, align 8
// CHECK-NEXT:   %x8 = alloca i64, align 8
// CHECK-NEXT:   %x10 = alloca i64, align 8
// CHECK-NEXT:   %x12 = alloca i64, align 8
// CHECK-NEXT:   %x14 = alloca i64, align 8
// CHECK-NEXT:   %x16 = alloca i64, align 8
// CHECK-NEXT:   store i64 1, ptr %x, align 8
// CHECK-NEXT:   %0 = load i64, ptr %x, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i64 2, ptr %x1, align 8
// CHECK-NEXT:   %1 = load i64, ptr %x1, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i64 3, ptr %x2, align 8
// CHECK-NEXT:   %2 = load i64, ptr %x2, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i64 4, ptr %x4, align 8
// CHECK-NEXT:   %3 = load i64, ptr %x4, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end5
// CHECK: expand.end5:
// CHECK-NEXT:   store i64 5, ptr %x6, align 8
// CHECK-NEXT:   %4 = load i64, ptr %x6, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next7
// CHECK: expand.next7:
// CHECK-NEXT:   store i64 6, ptr %x8, align 8
// CHECK-NEXT:   %5 = load i64, ptr %x8, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end9
// CHECK: expand.end9:
// CHECK-NEXT:   store i64 7, ptr %x10, align 8
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i64 8, ptr %x12, align 8
// CHECK-NEXT:   br label %expand.end13
// CHECK: expand.end13:
// CHECK-NEXT:   store i64 9, ptr %x14, align 8
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   store i64 10, ptr %x16, align 8
// CHECK-NEXT:   br label %expand.end17
// CHECK: expand.end17:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZN2s1IlE2tfIlEEvv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i64, align 8
// CHECK-NEXT:   %x2 = alloca i64, align 8
// CHECK-NEXT:   %x3 = alloca i64, align 8
// CHECK-NEXT:   %x5 = alloca i64, align 8
// CHECK-NEXT:   %x7 = alloca i64, align 8
// CHECK-NEXT:   %x9 = alloca i64, align 8
// CHECK-NEXT:   %x11 = alloca i64, align 8
// CHECK-NEXT:   %x13 = alloca i64, align 8
// CHECK-NEXT:   %x15 = alloca i64, align 8
// CHECK-NEXT:   %x17 = alloca i64, align 8
// CHECK-NEXT:   %x19 = alloca i64, align 8
// CHECK-NEXT:   %x21 = alloca i64, align 8
// CHECK-NEXT:   %x23 = alloca i64, align 8
// CHECK-NEXT:   %x25 = alloca i64, align 8
// CHECK-NEXT:   %x27 = alloca i64, align 8
// CHECK-NEXT:   %x29 = alloca i64, align 8
// CHECK-NEXT:   %x31 = alloca i64, align 8
// CHECK-NEXT:   %x33 = alloca i64, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i64 1, ptr %x, align 8
// CHECK-NEXT:   %0 = load i64, ptr %x, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i64 2, ptr %x2, align 8
// CHECK-NEXT:   %1 = load i64, ptr %x2, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i64 3, ptr %x3, align 8
// CHECK-NEXT:   %2 = load i64, ptr %x3, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next4
// CHECK: expand.next4:
// CHECK-NEXT:   store i64 4, ptr %x5, align 8
// CHECK-NEXT:   %3 = load i64, ptr %x5, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end6
// CHECK: expand.end6:
// CHECK-NEXT:   store i64 5, ptr %x7, align 8
// CHECK-NEXT:   %4 = load i64, ptr %x7, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i64 6, ptr %x9, align 8
// CHECK-NEXT:   %5 = load i64, ptr %x9, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end10
// CHECK: expand.end10:
// CHECK-NEXT:   store i64 7, ptr %x11, align 8
// CHECK-NEXT:   %6 = load i64, ptr %x11, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next12
// CHECK: expand.next12:
// CHECK-NEXT:   store i64 8, ptr %x13, align 8
// CHECK-NEXT:   %7 = load i64, ptr %x13, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   store i64 9, ptr %x15, align 8
// CHECK-NEXT:   %8 = load i64, ptr %x15, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next16
// CHECK: expand.next16:
// CHECK-NEXT:   store i64 10, ptr %x17, align 8
// CHECK-NEXT:   %9 = load i64, ptr %x17, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %9)
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   store i64 11, ptr %x19, align 8
// CHECK-NEXT:   %10 = load i64, ptr %x19, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %10)
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i64 12, ptr %x21, align 8
// CHECK-NEXT:   %11 = load i64, ptr %x21, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %11)
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   store i64 13, ptr %x23, align 8
// CHECK-NEXT:   br label %expand.next24
// CHECK: expand.next24:
// CHECK-NEXT:   store i64 14, ptr %x25, align 8
// CHECK-NEXT:   br label %expand.end26
// CHECK: expand.end26:
// CHECK-NEXT:   store i64 15, ptr %x27, align 8
// CHECK-NEXT:   br label %expand.next28
// CHECK: expand.next28:
// CHECK-NEXT:   store i64 16, ptr %x29, align 8
// CHECK-NEXT:   br label %expand.end30
// CHECK: expand.end30:
// CHECK-NEXT:   store i64 17, ptr %x31, align 8
// CHECK-NEXT:   br label %expand.next32
// CHECK: expand.next32:
// CHECK-NEXT:   store i64 18, ptr %x33, align 8
// CHECK-NEXT:   br label %expand.end34
// CHECK: expand.end34:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZN2s1IiE2tfIiEEvv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %x5 = alloca i32, align 4
// CHECK-NEXT:   %x7 = alloca i32, align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK-NEXT:   %x11 = alloca i32, align 4
// CHECK-NEXT:   %x13 = alloca i32, align 4
// CHECK-NEXT:   %x15 = alloca i32, align 4
// CHECK-NEXT:   %x17 = alloca i32, align 4
// CHECK-NEXT:   %x19 = alloca i32, align 4
// CHECK-NEXT:   %x21 = alloca i32, align 4
// CHECK-NEXT:   %x23 = alloca i32, align 4
// CHECK-NEXT:   %x25 = alloca i32, align 4
// CHECK-NEXT:   %x27 = alloca i32, align 4
// CHECK-NEXT:   %x29 = alloca i32, align 4
// CHECK-NEXT:   %x31 = alloca i32, align 4
// CHECK-NEXT:   %x33 = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x2, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x2, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 3, ptr %x3, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x3, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next4
// CHECK: expand.next4:
// CHECK-NEXT:   store i32 4, ptr %x5, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x5, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end6
// CHECK: expand.end6:
// CHECK-NEXT:   store i32 5, ptr %x7, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x7, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i32 6, ptr %x9, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x9, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end10
// CHECK: expand.end10:
// CHECK-NEXT:   store i32 7, ptr %x11, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x11, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next12
// CHECK: expand.next12:
// CHECK-NEXT:   store i32 8, ptr %x13, align 4
// CHECK-NEXT:   %7 = load i32, ptr %x13, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   store i32 9, ptr %x15, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x15, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next16
// CHECK: expand.next16:
// CHECK-NEXT:   store i32 10, ptr %x17, align 4
// CHECK-NEXT:   %9 = load i32, ptr %x17, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %9)
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   store i32 11, ptr %x19, align 4
// CHECK-NEXT:   %10 = load i32, ptr %x19, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %10)
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i32 12, ptr %x21, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x21, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %11)
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   store i32 13, ptr %x23, align 4
// CHECK-NEXT:   br label %expand.next24
// CHECK: expand.next24:
// CHECK-NEXT:   store i32 14, ptr %x25, align 4
// CHECK-NEXT:   br label %expand.end26
// CHECK: expand.end26:
// CHECK-NEXT:   store i32 15, ptr %x27, align 4
// CHECK-NEXT:   br label %expand.next28
// CHECK: expand.next28:
// CHECK-NEXT:   store i32 16, ptr %x29, align 4
// CHECK-NEXT:   br label %expand.end30
// CHECK: expand.end30:
// CHECK-NEXT:   store i32 17, ptr %x31, align 4
// CHECK-NEXT:   br label %expand.next32
// CHECK: expand.next32:
// CHECK-NEXT:   store i32 18, ptr %x33, align 4
// CHECK-NEXT:   br label %expand.end34
// CHECK: expand.end34:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZN2s1IiE2tfIlEEvv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i64, align 8
// CHECK-NEXT:   %x2 = alloca i64, align 8
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %x5 = alloca i32, align 4
// CHECK-NEXT:   %x7 = alloca i32, align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK-NEXT:   %x11 = alloca i64, align 8
// CHECK-NEXT:   %x13 = alloca i64, align 8
// CHECK-NEXT:   %x15 = alloca i64, align 8
// CHECK-NEXT:   %x17 = alloca i64, align 8
// CHECK-NEXT:   %x19 = alloca i32, align 4
// CHECK-NEXT:   %x21 = alloca i64, align 8
// CHECK-NEXT:   %x23 = alloca i32, align 4
// CHECK-NEXT:   %x25 = alloca i32, align 4
// CHECK-NEXT:   %x27 = alloca i64, align 8
// CHECK-NEXT:   %x29 = alloca i64, align 8
// CHECK-NEXT:   %x31 = alloca i64, align 8
// CHECK-NEXT:   %x33 = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i64 1, ptr %x, align 8
// CHECK-NEXT:   %0 = load i64, ptr %x, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i64 2, ptr %x2, align 8
// CHECK-NEXT:   %1 = load i64, ptr %x2, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 3, ptr %x3, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x3, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next4
// CHECK: expand.next4:
// CHECK-NEXT:   store i32 4, ptr %x5, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x5, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end6
// CHECK: expand.end6:
// CHECK-NEXT:   store i32 5, ptr %x7, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x7, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i32 6, ptr %x9, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x9, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end10
// CHECK: expand.end10:
// CHECK-NEXT:   store i64 7, ptr %x11, align 8
// CHECK-NEXT:   %6 = load i64, ptr %x11, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next12
// CHECK: expand.next12:
// CHECK-NEXT:   store i64 8, ptr %x13, align 8
// CHECK-NEXT:   %7 = load i64, ptr %x13, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   store i64 9, ptr %x15, align 8
// CHECK-NEXT:   %8 = load i64, ptr %x15, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next16
// CHECK: expand.next16:
// CHECK-NEXT:   store i64 10, ptr %x17, align 8
// CHECK-NEXT:   %9 = load i64, ptr %x17, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %9)
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   store i32 11, ptr %x19, align 4
// CHECK-NEXT:   %10 = load i32, ptr %x19, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %10)
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i64 12, ptr %x21, align 8
// CHECK-NEXT:   %11 = load i64, ptr %x21, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %11)
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   store i32 13, ptr %x23, align 4
// CHECK-NEXT:   br label %expand.next24
// CHECK: expand.next24:
// CHECK-NEXT:   store i32 14, ptr %x25, align 4
// CHECK-NEXT:   br label %expand.end26
// CHECK: expand.end26:
// CHECK-NEXT:   store i64 15, ptr %x27, align 8
// CHECK-NEXT:   br label %expand.next28
// CHECK: expand.next28:
// CHECK-NEXT:   store i64 16, ptr %x29, align 8
// CHECK-NEXT:   br label %expand.end30
// CHECK: expand.end30:
// CHECK-NEXT:   store i64 17, ptr %x31, align 8
// CHECK-NEXT:   br label %expand.next32
// CHECK: expand.next32:
// CHECK-NEXT:   store i32 18, ptr %x33, align 4
// CHECK-NEXT:   br label %expand.end34
// CHECK: expand.end34:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZN2s1IlE2tfIiEEvv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x3 = alloca i64, align 8
// CHECK-NEXT:   %x5 = alloca i64, align 8
// CHECK-NEXT:   %x7 = alloca i64, align 8
// CHECK-NEXT:   %x9 = alloca i64, align 8
// CHECK-NEXT:   %x11 = alloca i32, align 4
// CHECK-NEXT:   %x13 = alloca i32, align 4
// CHECK-NEXT:   %x15 = alloca i32, align 4
// CHECK-NEXT:   %x17 = alloca i32, align 4
// CHECK-NEXT:   %x19 = alloca i64, align 8
// CHECK-NEXT:   %x21 = alloca i32, align 4
// CHECK-NEXT:   %x23 = alloca i64, align 8
// CHECK-NEXT:   %x25 = alloca i64, align 8
// CHECK-NEXT:   %x27 = alloca i32, align 4
// CHECK-NEXT:   %x29 = alloca i32, align 4
// CHECK-NEXT:   %x31 = alloca i32, align 4
// CHECK-NEXT:   %x33 = alloca i64, align 8
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %0)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x2, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x2, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i64 3, ptr %x3, align 8
// CHECK-NEXT:   %2 = load i64, ptr %x3, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next4
// CHECK: expand.next4:
// CHECK-NEXT:   store i64 4, ptr %x5, align 8
// CHECK-NEXT:   %3 = load i64, ptr %x5, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %3)
// CHECK-NEXT:   br label %expand.end6
// CHECK: expand.end6:
// CHECK-NEXT:   store i64 5, ptr %x7, align 8
// CHECK-NEXT:   %4 = load i64, ptr %x7, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i64 6, ptr %x9, align 8
// CHECK-NEXT:   %5 = load i64, ptr %x9, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end10
// CHECK: expand.end10:
// CHECK-NEXT:   store i32 7, ptr %x11, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x11, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next12
// CHECK: expand.next12:
// CHECK-NEXT:   store i32 8, ptr %x13, align 4
// CHECK-NEXT:   %7 = load i32, ptr %x13, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   store i32 9, ptr %x15, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x15, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next16
// CHECK: expand.next16:
// CHECK-NEXT:   store i32 10, ptr %x17, align 4
// CHECK-NEXT:   %9 = load i32, ptr %x17, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %9)
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   store i64 11, ptr %x19, align 8
// CHECK-NEXT:   %10 = load i64, ptr %x19, align 8
// CHECK-NEXT:   call void @_Z1gl(i64 {{.*}} %10)
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i32 12, ptr %x21, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x21, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %11)
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   store i64 13, ptr %x23, align 8
// CHECK-NEXT:   br label %expand.next24
// CHECK: expand.next24:
// CHECK-NEXT:   store i64 14, ptr %x25, align 8
// CHECK-NEXT:   br label %expand.end26
// CHECK: expand.end26:
// CHECK-NEXT:   store i32 15, ptr %x27, align 4
// CHECK-NEXT:   br label %expand.next28
// CHECK: expand.next28:
// CHECK-NEXT:   store i32 16, ptr %x29, align 4
// CHECK-NEXT:   br label %expand.end30
// CHECK: expand.end30:
// CHECK-NEXT:   store i32 17, ptr %x31, align 4
// CHECK-NEXT:   br label %expand.next32
// CHECK: expand.next32:
// CHECK-NEXT:   store i64 18, ptr %x33, align 8
// CHECK-NEXT:   br label %expand.end34
// CHECK: expand.end34:
// CHECK-NEXT:   ret void
