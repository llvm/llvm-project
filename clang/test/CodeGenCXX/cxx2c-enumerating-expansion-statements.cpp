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

struct X {
  int a, b, c;
};

template <typename ...Ts>
void t3(Ts... ts) {
  template for (auto x : {ts...}) g(x);
  template for (auto x : {1, ts..., 2, ts..., 3}) g(x);
  template for (auto x : {4, ts..., ts..., 5}) g(x);
  template for (X x : {{ts...}, {ts...}, {6, 7, 8}}) g(x.a);
  template for (X x : {X{ts...}}) g(x.a);
}

template <int ...is>
void t4() {
  template for (constexpr auto x : {is...}) {
    g(x);
    tg<x>();
  }

  template for (constexpr auto x : {1, is..., 2, is..., 3}) {
    g(x);
    tg<x>();
  }

  template for (constexpr auto x : {4, is..., is..., 5}) {
    g(x);
    tg<x>();
  }

  template for (constexpr X x : {{is...}, {is...}, {6, 7, 8}}) {
    g(x.a);
    tg<x.a>();
  }

  template for (constexpr X x : {X{is...}}) {
    g(x.a);
    tg<x.a>();
  }
}

template <int ...is>
struct s2 {
  template <int ...js>
  void tf() {
    template for (auto x : {is..., js...}) g(x);
    template for (X x : {{is...}, {js...}}) g(x.a);
    template for (constexpr auto x : {is..., js...}) tg<x>();
    template for (constexpr X x : {{is...}, {js...}}) tg<x.a>();
  }
};

void f7() {
  t3(42, 43, 44);
  t4<42, 43, 44>();
  s2<1, 2, 3>().tf<4, 5, 6>();
}

template <int ...is>
void t5() {
  ([] {
    template for (constexpr auto x : {is}) {
      g(x);
      tg<x>();
    }
  }(), ...);
}

void f8() {
  t5<1, 2, 3>();
}

int references_enumerating() {
  int x = 1, y = 2, z = 3;
  template for (auto& v : {x, y, z}) { ++v; }
  template for (auto&& v : {x, y, z}) { ++v; }
  return x + y + z;
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


// CHECK-LABEL: define {{.*}} void @_Z2f7v()
// CHECK: entry:
// CHECK-NEXT:   %ref.tmp = alloca %struct.s2, align 1
// CHECK-NEXT:   call void @_Z2t3IJiiiEEvDpT_(i32 {{.*}} 42, i32 {{.*}} 43, i32 {{.*}} 44)
// CHECK-NEXT:   call void @_Z2t4IJLi42ELi43ELi44EEEvv()
// CHECK-NEXT:   call void @_ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   ret void

// CHECK-LABEL: define {{.*}} void @_Z2t3IJiiiEEvDpT_(i32 {{.*}} %ts, i32 {{.*}} %ts1, i32 {{.*}} %ts3)
// CHECK: entry:
// CHECK-NEXT:   %ts.addr = alloca i32, align 4
// CHECK-NEXT:   %ts.addr2 = alloca i32, align 4
// CHECK-NEXT:   %ts.addr4 = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x5 = alloca i32, align 4
// CHECK-NEXT:   %x7 = alloca i32, align 4
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
// CHECK-NEXT:   %x28 = alloca i32, align 4
// CHECK-NEXT:   %x30 = alloca i32, align 4
// CHECK-NEXT:   %x32 = alloca i32, align 4
// CHECK-NEXT:   %x34 = alloca i32, align 4
// CHECK-NEXT:   %x36 = alloca i32, align 4
// CHECK-NEXT:   %x38 = alloca i32, align 4
// CHECK-NEXT:   %x40 = alloca i32, align 4
// CHECK-NEXT:   %x42 = alloca %struct.X, align 4
// CHECK-NEXT:   %x45 = alloca %struct.X, align 4
// CHECK-NEXT:   %x51 = alloca %struct.X, align 4
// CHECK-NEXT:   %x54 = alloca %struct.X, align 4
// CHECK-NEXT:   store i32 %ts, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %ts1, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %ts3, ptr %ts.addr4, align 4
// CHECK-NEXT:   %0 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %0, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %1)
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %2 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %2, ptr %x5, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x5, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.next6
// CHECK: expand.next6:
// CHECK-NEXT:   %4 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %4, ptr %x7, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x7, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 1, ptr %x8, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x8, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   %7 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %7, ptr %x10, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x10, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %8)
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   %9 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %9, ptr %x12, align 4
// CHECK-NEXT:   %10 = load i32, ptr %x12, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %10)
// CHECK-NEXT:   br label %expand.next13
// CHECK: expand.next13:
// CHECK-NEXT:   %11 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %11, ptr %x14, align 4
// CHECK-NEXT:   %12 = load i32, ptr %x14, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %12)
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   store i32 2, ptr %x16, align 4
// CHECK-NEXT:   %13 = load i32, ptr %x16, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %13)
// CHECK-NEXT:   br label %expand.next17
// CHECK: expand.next17:
// CHECK-NEXT:   %14 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %14, ptr %x18, align 4
// CHECK-NEXT:   %15 = load i32, ptr %x18, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %15)
// CHECK-NEXT:   br label %expand.next19
// CHECK: expand.next19:
// CHECK-NEXT:   %16 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %16, ptr %x20, align 4
// CHECK-NEXT:   %17 = load i32, ptr %x20, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %17)
// CHECK-NEXT:   br label %expand.next21
// CHECK: expand.next21:
// CHECK-NEXT:   %18 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %18, ptr %x22, align 4
// CHECK-NEXT:   %19 = load i32, ptr %x22, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %19)
// CHECK-NEXT:   br label %expand.next23
// CHECK: expand.next23:
// CHECK-NEXT:   store i32 3, ptr %x24, align 4
// CHECK-NEXT:   %20 = load i32, ptr %x24, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %20)
// CHECK-NEXT:   br label %expand.end25
// CHECK: expand.end25:
// CHECK-NEXT:   store i32 4, ptr %x26, align 4
// CHECK-NEXT:   %21 = load i32, ptr %x26, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %21)
// CHECK-NEXT:   br label %expand.next27
// CHECK: expand.next27:
// CHECK-NEXT:   %22 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %22, ptr %x28, align 4
// CHECK-NEXT:   %23 = load i32, ptr %x28, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %23)
// CHECK-NEXT:   br label %expand.next29
// CHECK: expand.next29:
// CHECK-NEXT:   %24 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %24, ptr %x30, align 4
// CHECK-NEXT:   %25 = load i32, ptr %x30, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %25)
// CHECK-NEXT:   br label %expand.next31
// CHECK: expand.next31:
// CHECK-NEXT:   %26 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %26, ptr %x32, align 4
// CHECK-NEXT:   %27 = load i32, ptr %x32, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %27)
// CHECK-NEXT:   br label %expand.next33
// CHECK: expand.next33:
// CHECK-NEXT:   %28 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %28, ptr %x34, align 4
// CHECK-NEXT:   %29 = load i32, ptr %x34, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %29)
// CHECK-NEXT:   br label %expand.next35
// CHECK: expand.next35:
// CHECK-NEXT:   %30 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %30, ptr %x36, align 4
// CHECK-NEXT:   %31 = load i32, ptr %x36, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %31)
// CHECK-NEXT:   br label %expand.next37
// CHECK: expand.next37:
// CHECK-NEXT:   %32 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %32, ptr %x38, align 4
// CHECK-NEXT:   %33 = load i32, ptr %x38, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %33)
// CHECK-NEXT:   br label %expand.next39
// CHECK: expand.next39:
// CHECK-NEXT:   store i32 5, ptr %x40, align 4
// CHECK-NEXT:   %34 = load i32, ptr %x40, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %34)
// CHECK-NEXT:   br label %expand.end41
// CHECK: expand.end41:
// CHECK-NEXT:   %a = getelementptr inbounds nuw %struct.X, ptr %x42, i32 0, i32 0
// CHECK-NEXT:   %35 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %35, ptr %a, align 4
// CHECK-NEXT:   %b = getelementptr inbounds nuw %struct.X, ptr %x42, i32 0, i32 1
// CHECK-NEXT:   %36 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %36, ptr %b, align 4
// CHECK-NEXT:   %c = getelementptr inbounds nuw %struct.X, ptr %x42, i32 0, i32 2
// CHECK-NEXT:   %37 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %37, ptr %c, align 4
// CHECK-NEXT:   %a43 = getelementptr inbounds nuw %struct.X, ptr %x42, i32 0, i32 0
// CHECK-NEXT:   %38 = load i32, ptr %a43, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %38)
// CHECK-NEXT:   br label %expand.next44
// CHECK: expand.next44:
// CHECK-NEXT:   %a46 = getelementptr inbounds nuw %struct.X, ptr %x45, i32 0, i32 0
// CHECK-NEXT:   %39 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %39, ptr %a46, align 4
// CHECK-NEXT:   %b47 = getelementptr inbounds nuw %struct.X, ptr %x45, i32 0, i32 1
// CHECK-NEXT:   %40 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %40, ptr %b47, align 4
// CHECK-NEXT:   %c48 = getelementptr inbounds nuw %struct.X, ptr %x45, i32 0, i32 2
// CHECK-NEXT:   %41 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %41, ptr %c48, align 4
// CHECK-NEXT:   %a49 = getelementptr inbounds nuw %struct.X, ptr %x45, i32 0, i32 0
// CHECK-NEXT:   %42 = load i32, ptr %a49, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %42)
// CHECK-NEXT:   br label %expand.next50
// CHECK: expand.next50:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x51, ptr align 4 @__const._Z2t3IJiiiEEvDpT_.x, i64 12, i1 false)
// CHECK-NEXT:   %a52 = getelementptr inbounds nuw %struct.X, ptr %x51, i32 0, i32 0
// CHECK-NEXT:   %43 = load i32, ptr %a52, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %43)
// CHECK-NEXT:   br label %expand.end53
// CHECK: expand.end53:
// CHECK-NEXT:   %a55 = getelementptr inbounds nuw %struct.X, ptr %x54, i32 0, i32 0
// CHECK-NEXT:   %44 = load i32, ptr %ts.addr, align 4
// CHECK-NEXT:   store i32 %44, ptr %a55, align 4
// CHECK-NEXT:   %b56 = getelementptr inbounds nuw %struct.X, ptr %x54, i32 0, i32 1
// CHECK-NEXT:   %45 = load i32, ptr %ts.addr2, align 4
// CHECK-NEXT:   store i32 %45, ptr %b56, align 4
// CHECK-NEXT:   %c57 = getelementptr inbounds nuw %struct.X, ptr %x54, i32 0, i32 2
// CHECK-NEXT:   %46 = load i32, ptr %ts.addr4, align 4
// CHECK-NEXT:   store i32 %46, ptr %c57, align 4
// CHECK-NEXT:   %a58 = getelementptr inbounds nuw %struct.X, ptr %x54, i32 0, i32 0
// CHECK-NEXT:   %47 = load i32, ptr %a58, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %47)
// CHECK-NEXT:   br label %expand.end59
// CHECK: expand.end59:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2t4IJLi42ELi43ELi44EEEvv()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   %x15 = alloca i32, align 4
// CHECK-NEXT:   %x18 = alloca i32, align 4
// CHECK-NEXT:   %x21 = alloca i32, align 4
// CHECK-NEXT:   %x24 = alloca i32, align 4
// CHECK-NEXT:   %x27 = alloca i32, align 4
// CHECK-NEXT:   %x30 = alloca i32, align 4
// CHECK-NEXT:   %x33 = alloca i32, align 4
// CHECK-NEXT:   %x36 = alloca i32, align 4
// CHECK-NEXT:   %x39 = alloca i32, align 4
// CHECK-NEXT:   %x42 = alloca i32, align 4
// CHECK-NEXT:   %x45 = alloca i32, align 4
// CHECK-NEXT:   %x48 = alloca i32, align 4
// CHECK-NEXT:   %x51 = alloca i32, align 4
// CHECK-NEXT:   %x54 = alloca i32, align 4
// CHECK-NEXT:   %x57 = alloca %struct.X, align 4
// CHECK-NEXT:   %x60 = alloca %struct.X, align 4
// CHECK-NEXT:   %x63 = alloca %struct.X, align 4
// CHECK-NEXT:   %x66 = alloca %struct.X, align 4
// CHECK-NEXT:   store i32 42, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 43, ptr %x1, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 43)
// CHECK-NEXT:   %call2 = call {{.*}} i32 @_Z2tgILi43EEiv()
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 44, ptr %x4, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 44)
// CHECK-NEXT:   %call5 = call {{.*}} i32 @_Z2tgILi44EEiv()
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store i32 1, ptr %x6, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 1)
// CHECK-NEXT:   %call7 = call {{.*}} i32 @_Z2tgILi1EEiv()
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store i32 42, ptr %x9, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call10 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store i32 43, ptr %x12, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 43)
// CHECK-NEXT:   %call13 = call {{.*}} i32 @_Z2tgILi43EEiv()
// CHECK-NEXT:   br label %expand.next14
// CHECK: expand.next14:
// CHECK-NEXT:   store i32 44, ptr %x15, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 44)
// CHECK-NEXT:   %call16 = call {{.*}} i32 @_Z2tgILi44EEiv()
// CHECK-NEXT:   br label %expand.next17
// CHECK: expand.next17:
// CHECK-NEXT:   store i32 2, ptr %x18, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 2)
// CHECK-NEXT:   %call19 = call {{.*}} i32 @_Z2tgILi2EEiv()
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i32 42, ptr %x21, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call22 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next23
// CHECK: expand.next23:
// CHECK-NEXT:   store i32 43, ptr %x24, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 43)
// CHECK-NEXT:   %call25 = call {{.*}} i32 @_Z2tgILi43EEiv()
// CHECK-NEXT:   br label %expand.next26
// CHECK: expand.next26:
// CHECK-NEXT:   store i32 44, ptr %x27, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 44)
// CHECK-NEXT:   %call28 = call {{.*}} i32 @_Z2tgILi44EEiv()
// CHECK-NEXT:   br label %expand.next29
// CHECK: expand.next29:
// CHECK-NEXT:   store i32 3, ptr %x30, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 3)
// CHECK-NEXT:   %call31 = call {{.*}} i32 @_Z2tgILi3EEiv()
// CHECK-NEXT:   br label %expand.end32
// CHECK: expand.end32:
// CHECK-NEXT:   store i32 4, ptr %x33, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 4)
// CHECK-NEXT:   %call34 = call {{.*}} i32 @_Z2tgILi4EEiv()
// CHECK-NEXT:   br label %expand.next35
// CHECK: expand.next35:
// CHECK-NEXT:   store i32 42, ptr %x36, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call37 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next38
// CHECK: expand.next38:
// CHECK-NEXT:   store i32 43, ptr %x39, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 43)
// CHECK-NEXT:   %call40 = call {{.*}} i32 @_Z2tgILi43EEiv()
// CHECK-NEXT:   br label %expand.next41
// CHECK: expand.next41:
// CHECK-NEXT:   store i32 44, ptr %x42, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 44)
// CHECK-NEXT:   %call43 = call {{.*}} i32 @_Z2tgILi44EEiv()
// CHECK-NEXT:   br label %expand.next44
// CHECK: expand.next44:
// CHECK-NEXT:   store i32 42, ptr %x45, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call46 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next47
// CHECK: expand.next47:
// CHECK-NEXT:   store i32 43, ptr %x48, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 43)
// CHECK-NEXT:   %call49 = call {{.*}} i32 @_Z2tgILi43EEiv()
// CHECK-NEXT:   br label %expand.next50
// CHECK: expand.next50:
// CHECK-NEXT:   store i32 44, ptr %x51, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 44)
// CHECK-NEXT:   %call52 = call {{.*}} i32 @_Z2tgILi44EEiv()
// CHECK-NEXT:   br label %expand.next53
// CHECK: expand.next53:
// CHECK-NEXT:   store i32 5, ptr %x54, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 5)
// CHECK-NEXT:   %call55 = call {{.*}} i32 @_Z2tgILi5EEiv()
// CHECK-NEXT:   br label %expand.end56
// CHECK: expand.end56:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x57, ptr align 4 @__const._Z2t4IJLi42ELi43ELi44EEEvv.x, i64 12, i1 false)
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call58 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next59
// CHECK: expand.next59:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x60, ptr align 4 @__const._Z2t4IJLi42ELi43ELi44EEEvv.x.3, i64 12, i1 false)
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call61 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.next62
// CHECK: expand.next62:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x63, ptr align 4 @__const._Z2t4IJLi42ELi43ELi44EEEvv.x.4, i64 12, i1 false)
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 6)
// CHECK-NEXT:   %call64 = call {{.*}} i32 @_Z2tgILi6EEiv()
// CHECK-NEXT:   br label %expand.end65
// CHECK: expand.end65:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x66, ptr align 4 @__const._Z2t4IJLi42ELi43ELi44EEEvv.x.5, i64 12, i1 false)
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 42)
// CHECK-NEXT:   %call67 = call {{.*}} i32 @_Z2tgILi42EEiv()
// CHECK-NEXT:   br label %expand.end68
// CHECK: expand.end68:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   %x10 = alloca i32, align 4
// CHECK-NEXT:   %x11 = alloca %struct.X, align 4
// CHECK-NEXT:   %x13 = alloca %struct.X, align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK-NEXT:   %x18 = alloca i32, align 4
// CHECK-NEXT:   %x21 = alloca i32, align 4
// CHECK-NEXT:   %x24 = alloca i32, align 4
// CHECK-NEXT:   %x27 = alloca i32, align 4
// CHECK-NEXT:   %x30 = alloca i32, align 4
// CHECK-NEXT:   %x33 = alloca %struct.X, align 4
// CHECK-NEXT:   %x36 = alloca %struct.X, align 4
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
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 3, ptr %x4, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x4, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %2)
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   store i32 4, ptr %x6, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x6, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %3)
// CHECK-NEXT:   br label %expand.next7
// CHECK: expand.next7:
// CHECK-NEXT:   store i32 5, ptr %x8, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x8, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %4)
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   store i32 6, ptr %x10, align 4
// CHECK-NEXT:   %5 = load i32, ptr %x10, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %5)
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x11, ptr align 4 @__const._ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv.x, i64 12, i1 false)
// CHECK-NEXT:   %a = getelementptr inbounds nuw %struct.X, ptr %x11, i32 0, i32 0
// CHECK-NEXT:   %6 = load i32, ptr %a, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %6)
// CHECK-NEXT:   br label %expand.next12
// CHECK: expand.next12:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x13, ptr align 4 @__const._ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv.x.6, i64 12, i1 false)
// CHECK-NEXT:   %a14 = getelementptr inbounds nuw %struct.X, ptr %x13, i32 0, i32 0
// CHECK-NEXT:   %7 = load i32, ptr %a14, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} %7)
// CHECK-NEXT:   br label %expand.end15
// CHECK: expand.end15:
// CHECK-NEXT:   store i32 1, ptr %x16, align 4
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi1EEiv()
// CHECK-NEXT:   br label %expand.next17
// CHECK: expand.next17:
// CHECK-NEXT:   store i32 2, ptr %x18, align 4
// CHECK-NEXT:   %call19 = call {{.*}} i32 @_Z2tgILi2EEiv()
// CHECK-NEXT:   br label %expand.next20
// CHECK: expand.next20:
// CHECK-NEXT:   store i32 3, ptr %x21, align 4
// CHECK-NEXT:   %call22 = call {{.*}} i32 @_Z2tgILi3EEiv()
// CHECK-NEXT:   br label %expand.next23
// CHECK: expand.next23:
// CHECK-NEXT:   store i32 4, ptr %x24, align 4
// CHECK-NEXT:   %call25 = call {{.*}} i32 @_Z2tgILi4EEiv()
// CHECK-NEXT:   br label %expand.next26
// CHECK: expand.next26:
// CHECK-NEXT:   store i32 5, ptr %x27, align 4
// CHECK-NEXT:   %call28 = call {{.*}} i32 @_Z2tgILi5EEiv()
// CHECK-NEXT:   br label %expand.next29
// CHECK: expand.next29:
// CHECK-NEXT:   store i32 6, ptr %x30, align 4
// CHECK-NEXT:   %call31 = call {{.*}} i32 @_Z2tgILi6EEiv()
// CHECK-NEXT:   br label %expand.end32
// CHECK: expand.end32:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x33, ptr align 4 @__const._ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv.x.7, i64 12, i1 false)
// CHECK-NEXT:   %call34 = call {{.*}} i32 @_Z2tgILi1EEiv()
// CHECK-NEXT:   br label %expand.next35
// CHECK: expand.next35:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x36, ptr align 4 @__const._ZN2s2IJLi1ELi2ELi3EEE2tfIJLi4ELi5ELi6EEEEvv.x.8, i64 12, i1 false)
// CHECK-NEXT:   %call37 = call {{.*}} i32 @_Z2tgILi4EEiv()
// CHECK-NEXT:   br label %expand.end38
// CHECK: expand.end38:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2f8v()
// CHECK: entry:
// CHECK-NEXT:   call void @_Z2t5IJLi1ELi2ELi3EEEvv()
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_Z2t5IJLi1ELi2ELi3EEEvv()
// CHECK: entry:
// CHECK-NEXT:   %ref.tmp = alloca %class.anon, align 1
// CHECK-NEXT:   %ref.tmp1 = alloca %class.anon.1, align 1
// CHECK-NEXT:   %ref.tmp2 = alloca %class.anon.3, align 1
// CHECK-NEXT:   call void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE1_clEv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   call void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE0_clEv(ptr {{.*}} %ref.tmp1)
// CHECK-NEXT:   call void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE_clEv(ptr {{.*}} %ref.tmp2)
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} i32 @_Z22references_enumeratingv()
// CHECK: entry:
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %z = alloca i32, align 4
// CHECK-NEXT:   %v = alloca ptr, align 8
// CHECK-NEXT:   %v1 = alloca ptr, align 8
// CHECK-NEXT:   %v4 = alloca ptr, align 8
// CHECK-NEXT:   %v6 = alloca ptr, align 8
// CHECK-NEXT:   %v9 = alloca ptr, align 8
// CHECK-NEXT:   %v12 = alloca ptr, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   store i32 2, ptr %y, align 4
// CHECK-NEXT:   store i32 3, ptr %z, align 4
// CHECK-NEXT:   store ptr %x, ptr %v, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %v, align 8
// CHECK-NEXT:   %1 = load i32, ptr %0, align 4
// CHECK-NEXT:   %inc = add nsw i32 %1, 1
// CHECK-NEXT:   store i32 %inc, ptr %0, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store ptr %y, ptr %v1, align 8
// CHECK-NEXT:   %2 = load ptr, ptr %v1, align 8
// CHECK-NEXT:   %3 = load i32, ptr %2, align 4
// CHECK-NEXT:   %inc2 = add nsw i32 %3, 1
// CHECK-NEXT:   store i32 %inc2, ptr %2, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store ptr %z, ptr %v4, align 8
// CHECK-NEXT:   %4 = load ptr, ptr %v4, align 8
// CHECK-NEXT:   %5 = load i32, ptr %4, align 4
// CHECK-NEXT:   %inc5 = add nsw i32 %5, 1
// CHECK-NEXT:   store i32 %inc5, ptr %4, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr %x, ptr %v6, align 8
// CHECK-NEXT:   %6 = load ptr, ptr %v6, align 8
// CHECK-NEXT:   %7 = load i32, ptr %6, align 4
// CHECK-NEXT:   %inc7 = add nsw i32 %7, 1
// CHECK-NEXT:   store i32 %inc7, ptr %6, align 4
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   store ptr %y, ptr %v9, align 8
// CHECK-NEXT:   %8 = load ptr, ptr %v9, align 8
// CHECK-NEXT:   %9 = load i32, ptr %8, align 4
// CHECK-NEXT:   %inc10 = add nsw i32 %9, 1
// CHECK-NEXT:   store i32 %inc10, ptr %8, align 4
// CHECK-NEXT:   br label %expand.next11
// CHECK: expand.next11:
// CHECK-NEXT:   store ptr %z, ptr %v12, align 8
// CHECK-NEXT:   %10 = load ptr, ptr %v12, align 8
// CHECK-NEXT:   %11 = load i32, ptr %10, align 4
// CHECK-NEXT:   %inc13 = add nsw i32 %11, 1
// CHECK-NEXT:   store i32 %inc13, ptr %10, align 4
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   %12 = load i32, ptr %x, align 4
// CHECK-NEXT:   %13 = load i32, ptr %y, align 4
// CHECK-NEXT:   %add = add nsw i32 %12, %13
// CHECK-NEXT:   %14 = load i32, ptr %z, align 4
// CHECK-NEXT:   %add15 = add nsw i32 %add, %14
// CHECK-NEXT:   ret i32 %add15


// CHECK-LABEL: define {{.*}} void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE1_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 1)
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi1EEiv()
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE0_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 2, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 2)
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi2EEiv()
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void


// CHECK-LABEL: define {{.*}} void @_ZZ2t5IJLi1ELi2ELi3EEEvvENKUlvE_clEv(ptr {{.*}} %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT:   store i32 3, ptr %x, align 4
// CHECK-NEXT:   call void @_Z1gi(i32 {{.*}} 3)
// CHECK-NEXT:   %call = call {{.*}} i32 @_Z2tgILi3EEiv()
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   ret void
