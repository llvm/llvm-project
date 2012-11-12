// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s

// CHECK-NOT: @unused
auto unused = [](int i) { return i+1; };

// CHECK: @used = internal global
auto used = [](int i) { return i+1; };
void *use = &used;

// CHECK: @cvar = global
extern "C" auto cvar = []{};

int a() { return []{ return 1; }(); }
// CHECK: define i32 @_Z1av
// CHECK: call i32 @"_ZZ1avENK3$_0clEv"
// CHECK: define internal i32 @"_ZZ1avENK3$_0clEv"
// CHECK: ret i32 1

int b(int x) { return [x]{return x;}(); }
// CHECK: define i32 @_Z1bi
// CHECK: store i32
// CHECK: load i32*
// CHECK: store i32
// CHECK: call i32 @"_ZZ1biENK3$_1clEv"
// CHECK: define internal i32 @"_ZZ1biENK3$_1clEv"
// CHECK: load i32*
// CHECK: ret i32

int c(int x) { return [&x]{return x;}(); }
// CHECK: define i32 @_Z1ci
// CHECK: store i32
// CHECK: store i32*
// CHECK: call i32 @"_ZZ1ciENK3$_2clEv"
// CHECK: define internal i32 @"_ZZ1ciENK3$_2clEv"
// CHECK: load i32**
// CHECK: load i32*
// CHECK: ret i32

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; [x,y] { return y[x].x; }(); }

// CHECK: define i32 @_Z1di
// CHECK: call void @_ZN1DC1Ev
// CHECK: icmp ult i64 %{{.*}}, 10
// CHECK: call void @_ZN1DC1ERKS_
// CHECK: call i32 @"_ZZ1diENK3$_3clEv"
// CHECK: define internal i32 @"_ZZ1diENK3$_3clEv"
// CHECK: load i32*
// CHECK: load i32*
// CHECK: ret i32

struct E { E(); E(const E&); ~E(); int x; };
int e(E a, E b, bool cond) { [a,b,cond](){ return (cond ? a : b).x; }(); }
// CHECK: define i32 @_Z1e1ES_b
// CHECK: call void @_ZN1EC1ERKS_
// CHECK: invoke void @_ZN1EC1ERKS_
// CHECK: invoke i32 @"_ZZ1e1ES_bENK3$_4clEv"
// CHECK: call void @"_ZZ1e1ES_bEN3$_4D1Ev"
// CHECK: call void @"_ZZ1e1ES_bEN3$_4D1Ev"

// CHECK: define internal i32 @"_ZZ1e1ES_bENK3$_4clEv"
// CHECK: trunc i8
// CHECK: load i32*
// CHECK: ret i32

void f() {
  // CHECK: define void @_Z1fv()
  // CHECK: @"_ZZ1fvENK3$_5cvPFiiiEEv"
  // CHECK-NEXT: store i32 (i32, i32)*
  // CHECK-NEXT: ret void
  int (*fp)(int, int) = [](int x, int y){ return x + y; };
}

static int k;
int g() {
  int &r = k;
  // CHECK: define internal i32 @"_ZZ1gvENK3$_6clEv"(
  // CHECK-NOT: }
  // CHECK: load i32* @_ZL1k,
  return [] { return r; } ();
};

// CHECK: define internal i32 @"_ZZ1fvEN3$_58__invokeEii"
// CHECK: store i32
// CHECK-NEXT: store i32
// CHECK-NEXT: load i32*
// CHECK-NEXT: load i32*
// CHECK-NEXT: call i32 @"_ZZ1fvENK3$_5clEii"
// CHECK-NEXT: ret i32

// CHECK: define internal void @"_ZZ1e1ES_bEN3$_4D2Ev"
