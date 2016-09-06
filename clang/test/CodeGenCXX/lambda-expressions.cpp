// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s

// CHECK-NOT: @unused
auto unused = [](int i) { return i+1; };

// CHECK: @used = internal global
auto used = [](int i) { return i+1; };
void *use = &used;

// CHECK: @cvar = global
extern "C" auto cvar = []{};

// CHECK-LABEL: define i32 @_Z9ARBSizeOfi(i32
int ARBSizeOf(int n) {
  typedef double(T)[8][n];
  using TT = double[8][n];
  return [&]() -> int {
    typedef double(T1)[8][n];
    using TT1 = double[8][n];
    return [&n]() -> int {
      typedef double(T2)[8][n];
      using TT2 = double[8][n];
      return sizeof(T) + sizeof(T1) + sizeof(T2) + sizeof(TT) + sizeof(TT1) + sizeof(TT2);
    }();
  }();
}

// CHECK-LABEL: define internal i32 @"_ZZ9ARBSizeOfiENK3$_0clEv"

int a() { return []{ return 1; }(); }
// CHECK-LABEL: define i32 @_Z1av
// CHECK: call i32 @"_ZZ1avENK3$_1clEv"
// CHECK-LABEL: define internal i32 @"_ZZ1avENK3$_1clEv"
// CHECK: ret i32 1

int b(int x) { return [x]{return x;}(); }
// CHECK-LABEL: define i32 @_Z1bi
// CHECK: store i32
// CHECK: load i32, i32*
// CHECK: store i32
// CHECK: call i32 @"_ZZ1biENK3$_2clEv"
// CHECK-LABEL: define internal i32 @"_ZZ1biENK3$_2clEv"
// CHECK: load i32, i32*
// CHECK: ret i32

int c(int x) { return [&x]{return x;}(); }
// CHECK-LABEL: define i32 @_Z1ci
// CHECK: store i32
// CHECK: store i32*
// CHECK: call i32 @"_ZZ1ciENK3$_3clEv"
// CHECK-LABEL: define internal i32 @"_ZZ1ciENK3$_3clEv"
// CHECK: load i32*, i32**
// CHECK: load i32, i32*
// CHECK: ret i32

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; [x,y] { return y[x].x; }(); }

// CHECK-LABEL: define i32 @_Z1di
// CHECK: call void @_ZN1DC1Ev
// CHECK: icmp ult i64 %{{.*}}, 10
// CHECK: call void @_ZN1DC1ERKS_
// CHECK: call i32 @"_ZZ1diENK3$_4clEv"
// CHECK-LABEL: define internal i32 @"_ZZ1diENK3$_4clEv"
// CHECK: load i32, i32*
// CHECK: load i32, i32*
// CHECK: ret i32

struct E { E(); E(const E&); ~E(); int x; };
int e(E a, E b, bool cond) { [a,b,cond](){ return (cond ? a : b).x; }(); }
// CHECK-LABEL: define i32 @_Z1e1ES_b
// CHECK: call void @_ZN1EC1ERKS_
// CHECK: invoke void @_ZN1EC1ERKS_
// CHECK: invoke i32 @"_ZZ1e1ES_bENK3$_5clEv"
// CHECK: call void @"_ZZ1e1ES_bEN3$_5D1Ev"
// CHECK: call void @"_ZZ1e1ES_bEN3$_5D1Ev"

// CHECK-LABEL: define internal i32 @"_ZZ1e1ES_bENK3$_5clEv"
// CHECK: trunc i8
// CHECK: load i32, i32*
// CHECK: ret i32

void f() {
  // CHECK-LABEL: define void @_Z1fv()
  // CHECK: @"_ZZ1fvENK3$_6cvPFiiiEEv"
  // CHECK-NEXT: store i32 (i32, i32)*
  // CHECK-NEXT: ret void
  int (*fp)(int, int) = [](int x, int y){ return x + y; };
}

static int k;
int g() {
  int &r = k;
  // CHECK-LABEL: define internal i32 @"_ZZ1gvENK3$_7clEv"(
  // CHECK-NOT: }
  // CHECK: load i32, i32* @_ZL1k,
  return [] { return r; } ();
};

// PR14773
// CHECK: [[ARRVAL:%[0-9a-zA-Z]*]] = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @_ZZ14staticarrayrefvE5array, i64 0, i64 0), align 4
// CHECK-NEXT: store i32 [[ARRVAL]]
void staticarrayref(){
  static int array[] = {};
  (void)[](){
    int (&xxx)[0] = array;
    int y = xxx[0];
  }();
}

// CHECK-LABEL: define internal i32* @"_ZZ11PR22071_funvENK3$_9clEv"
// CHECK: ret i32* @PR22071_var
int PR22071_var;
int *PR22071_fun() {
  constexpr int &y = PR22071_var;
  return [&] { return &y; }();
}

namespace pr28595 {
  struct Temp {
    Temp();
    ~Temp() noexcept(false);
  };
  struct A {
    A();
    A(const A &a, const Temp &temp = Temp());
    ~A();
  };

  // CHECK-LABEL: define void @_ZN7pr285954testEv()
  void test() {
    // CHECK: [[ARRAY:%.*]] = alloca [3 x [5 x [[A:%.*]]]], align 1
    // CHECK: [[DESTIDX:%.*]] = alloca i64, align 8
    // CHECK: [[I0:%.*]] = alloca i64, align 8
    // CHECK: [[I1:%.*]] = alloca i64, align 8
    A array[3][5];

    // CHECK: [[DESTBASE:%.*]] = bitcast [3 x [5 x [[A]]]]* {{.*}} to [[A]]*
    // CHECK: store i64 0, i64* [[DESTIDX]], align 8
    // CHECK: store i64 0, i64* [[I0]], align 8
    // CHECK: br label
    // CHECK: icmp ult
    // CHECK: store i64 0, i64* [[I1]], align 8
    // CHECK: br label
    // CHECK: icmp ult
    // CHECK: [[T0:%.*]] = load i64, i64* [[DESTIDX]], align 8
    // CHECK: [[DEST:%.*]] = getelementptr inbounds [[A]], [[A]]* [[DESTBASE]], i64 [[T0]]
    // CHECK: invoke void @_ZN7pr285954TempC1Ev
    // CHECK: invoke void @_ZN7pr285951AC1ERKS0_RKNS_4TempE
    // CHECK: invoke void @_ZN7pr285954TempD1Ev
    // CHECK: landingpad
    // CHECK: landingpad
    // CHECK: br label [[CLEANUP:%.*]]{{$}}
    // CHECK: landingpad
    // CHECK: invoke void @_ZN7pr285954TempD1Ev
    // CHECK: br label [[CLEANUP]]
    // CHECK: icmp eq [[A]]* [[DESTBASE]], [[DEST]]
    // CHECK: [[T0:%.*]] = phi [[A]]*
    // CHECK: [[T1:%.*]] = getelementptr inbounds [[A]], [[A]]* [[T0]], i64 -1
    // CHECK: call void @_ZN7pr285951AD1Ev([[A]]* [[T1]])
    // CHECK: icmp eq [[A]]* [[T1]], [[DESTBASE]]
    (void) [array]{};

    //   Skip over the initialization loop.
    // CHECK: [[BEGIN:%.*]] = getelementptr inbounds [3 x [5 x [[A]]]], [3 x [5 x [[A]]]]* [[ARRAY]], i32 0, i32 0, i32 0
  }
}

// CHECK-LABEL: define internal void @"_ZZ1e1ES_bEN3$_5D2Ev"

// CHECK-LABEL: define internal i32 @"_ZZ1fvEN3$_68__invokeEii"
// CHECK: store i32
// CHECK-NEXT: store i32
// CHECK-NEXT: load i32, i32*
// CHECK-NEXT: load i32, i32*
// CHECK-NEXT: call i32 @"_ZZ1fvENK3$_6clEii"
// CHECK-NEXT: ret i32

// CHECK-LABEL: define internal void @"_ZZ1hvEN4$_118__invokeEv"(%struct.A* noalias sret %agg.result) {{.*}} {
// CHECK-NOT: =
// CHECK: call void @"_ZZ1hvENK4$_11clEv"(%struct.A* sret %agg.result,
// CHECK-NEXT: ret void
struct A { ~A(); };
void h() {
  A (*h)() = [] { return A(); };
}

// <rdar://problem/12778708>
struct XXX {};
void nestedCapture () {
  XXX localKey;
  ^() {
    [&]() {
      ^{ XXX k = localKey; };
    };
  };
}

// Ensure we don't assert here.
struct CaptureArrayAndThis {
  CaptureArrayAndThis() {
    char array[] = "floop";
    [array, this] {};
  }
} capture_array_and_this;

