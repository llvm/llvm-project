// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s

// CHECK-NOT: @unused
auto unused = [](int i) { return i+1; };

// CHECK: @used = internal global
auto used = [](int i) { return i+1; };
void *use = &used;

// CHECK: @cvar ={{.*}} global
extern "C" auto cvar = []{};

// CHECK-LABEL: define{{.*}} i32 @_Z9ARBSizeOfi(i32
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

// CHECK-LABEL: define internal noundef i32 @"_ZZ9ARBSizeOfiENK3$_0clEv"

int a() { return []{ return 1; }(); }
// CHECK-LABEL: define{{.*}} i32 @_Z1av
// CHECK: call noundef i32 @"_ZZ1avENK3$_0clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZZ1avENK3$_0clEv"
// CHECK: ret i32 1

int b(int x) { return [x]{return x;}(); }
// CHECK-LABEL: define{{.*}} i32 @_Z1bi
// CHECK: store i32
// CHECK: load i32, ptr
// CHECK: store i32
// CHECK: call noundef i32 @"_ZZ1biENK3$_0clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZZ1biENK3$_0clEv"
// CHECK: load i32, ptr
// CHECK: ret i32

int c(int x) { return [&x]{return x;}(); }
// CHECK-LABEL: define{{.*}} i32 @_Z1ci
// CHECK: store i32
// CHECK: store ptr
// CHECK: call noundef i32 @"_ZZ1ciENK3$_0clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZZ1ciENK3$_0clEv"
// CHECK: load ptr, ptr
// CHECK: load i32, ptr
// CHECK: ret i32

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; return [x,y] { return y[x].x; }(); }

// CHECK-LABEL: define{{.*}} i32 @_Z1di
// CHECK: call void @_ZN1DC1Ev
// CHECK: br label
// CHECK: call void @_ZN1DC1ERKS_
// CHECK: icmp eq i64 %{{.*}}, 10
// CHECK: br i1
// CHECK: call noundef i32 @"_ZZ1diENK3$_0clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZZ1diENK3$_0clEv"
// CHECK: load i32, ptr
// CHECK: load i32, ptr
// CHECK: ret i32

struct E { E(); E(const E&); ~E(); int x; };
int e(E a, E b, bool cond) { return [a,b,cond](){ return (cond ? a : b).x; }(); }
// CHECK-LABEL: define{{.*}} i32 @_Z1e1ES_b
// CHECK: call void @_ZN1EC1ERKS_
// CHECK: invoke void @_ZN1EC1ERKS_
// CHECK: invoke noundef i32 @"_ZZ1e1ES_bENK3$_0clEv"
// CHECK: call void @"_ZZ1e1ES_bEN3$_0D1Ev"
// CHECK: call void @"_ZZ1e1ES_bEN3$_0D1Ev"

// CHECK-LABEL: define internal noundef i32 @"_ZZ1e1ES_bENK3$_0clEv"
// CHECK: trunc i8
// CHECK: load i32, ptr
// CHECK: ret i32

void f() {
  // CHECK-LABEL: define{{.*}} void @_Z1fv()
  // CHECK: @"_ZZ1fvENK3$_0cvPFiiiEEv"
  // CHECK-NEXT: store ptr
  // CHECK-NEXT: ret void
  int (*fp)(int, int) = [](int x, int y){ return x + y; };
}

static int k;
int g() {
  int &r = k;
  // CHECK-LABEL: define internal noundef i32 @"_ZZ1gvENK3$_0clEv"(
  // CHECK-NOT: }
  // CHECK: load i32, ptr @_ZL1k,
  return [] { return r; } ();
};

// PR14773
// CHECK: [[ARRVAL:%[0-9a-zA-Z]*]] = load i32, ptr @_ZZ14staticarrayrefvE5array, align 4
// CHECK-NEXT: store i32 [[ARRVAL]]
void staticarrayref(){
  static int array[] = {};
  (void)[](){
    int (&xxx)[0] = array;
    int y = xxx[0];
  }();
}

// CHECK-LABEL: define internal noundef ptr @"_ZZ11PR22071_funvENK3$_0clEv"
// CHECK: ret ptr @PR22071_var
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

  void after_init() noexcept;

  // CHECK-LABEL: define{{.*}} void @_ZN7pr285954testEv()
  void test() {
    // CHECK: %[[SRC:.*]] = alloca [3 x [5 x %[[A:.*]]]], align 1
    A array[3][5];

    // Skip over the initialization loop.
    // CHECK: call {{.*}}after_init
    after_init();

    // CHECK: %[[DST_0:.*]] = getelementptr {{.*}} ptr %[[DST:.*]], i64 0, i64 0
    // CHECK: br label
    // CHECK: %[[I:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[I_NEXT:.*]], {{.*}} ]
    // CHECK: %[[DST_I:.*]] = getelementptr {{.*}} ptr %[[DST_0]], i64 %[[I]]
    // CHECK: %[[SRC_I:.*]] = getelementptr {{.*}} ptr %[[SRC]], i64 0, i64 %[[I]]
    //
    // CHECK: %[[DST_I_0:.*]] = getelementptr {{.*}} ptr %[[DST_I]], i64 0, i64 0
    // CHECK: br label
    // CHECK: %[[J:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[J_NEXT:.*]], {{.*}} ]
    // CHECK: %[[DST_I_J:.*]] = getelementptr {{.*}} ptr %[[DST_I_0]], i64 %[[J]]
    // CHECK: %[[SRC_I_J:.*]] = getelementptr {{.*}} ptr %[[SRC_I]], i64 0, i64 %[[J]]
    //
    // CHECK: invoke void @_ZN7pr285954TempC1Ev
    // CHECK: invoke void @_ZN7pr285951AC1ERKS0_RKNS_4TempE
    // CHECK: invoke void @_ZN7pr285954TempD1Ev
    //
    // CHECK: add nuw i64 %[[J]], 1
    // CHECK: icmp eq
    // CHECK: br i1
    //
    // CHECK: add nuw i64 %[[I]], 1
    // CHECK: icmp eq
    // CHECK: br i1
    //
    // CHECK: ret void
    //
    // CHECK: landingpad
    // CHECK: landingpad
    // CHECK: br label %[[CLEANUP:.*]]{{$}}
    // CHECK: landingpad
    // CHECK: invoke void @_ZN7pr285954TempD1Ev
    // CHECK: br label %[[CLEANUP]]
    //
    // CHECK: [[CLEANUP]]:
    // CHECK: icmp eq ptr %[[DST_0]], %[[DST_I_J]]
    // CHECK: %[[T0:.*]] = phi ptr
    // CHECK: %[[T1:.*]] = getelementptr inbounds %[[A]], ptr %[[T0]], i64 -1
    // CHECK: call void @_ZN7pr285951AD1Ev(ptr {{[^,]*}} %[[T1]])
    // CHECK: icmp eq ptr %[[T1]], %[[DST_0]]
    (void) [array]{};
  }
}

// CHECK-LABEL: define internal void @"_ZZ1e1ES_bEN3$_0D2Ev"

// CHECK-LABEL: define internal noundef i32 @"_ZZ1fvEN3$_08__invokeEii"
// CHECK: store i32
// CHECK-NEXT: store i32
// CHECK-NEXT: load i32, ptr
// CHECK-NEXT: load i32, ptr
// CHECK-NEXT: call noundef i32 @"_ZZ1fvENK3$_0clEii"
// CHECK-NEXT: ret i32

// CHECK-LABEL: define internal void @"_ZZ1hvEN3$_08__invokeEv"(ptr noalias sret(%struct.A) align 1 %agg.result) {{.*}} {
// CHECK: call void @"_ZZ1hvENK3$_0clEv"(ptr sret(%struct.A) align 1 %agg.result,
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

