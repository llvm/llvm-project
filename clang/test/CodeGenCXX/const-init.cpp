// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -std=c++98 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -std=c++11 -o - %s | FileCheck %s

// CHECK: @a = dso_local global i32 10
int a = 10;
// CHECK: @ar = dso_local constant i32* @a
int &ar = a;

void f();
// CHECK: @fr = dso_local constant void ()* @_Z1fv
void (&fr)() = f;

struct S { int& a; };
// CHECK: @s = dso_local global %struct.S { i32* @a }
S s = { a };

// PR5581
namespace PR5581 {
class C {
public:
  enum { e0, e1 };
  unsigned f;
};

// CHECK: @_ZN6PR55812g0E = dso_local global %"class.PR5581::C" { i32 1 }
C g0 = { C::e1 };
}

namespace test2 {
  struct A {
#if __cplusplus <= 199711L
    static const double d = 1.0;
    static const float f = d / 2;
#else
    static constexpr double d = 1.0;
    static constexpr float f = d / 2;
#endif
    static int g();
  } a;

  // CHECK: @_ZN5test22t0E = dso_local global double {{1\.0+e\+0+}}, align 8
  // CHECK: @_ZN5test22t1E = dso_local global [2 x double] [double {{1\.0+e\+0+}}, double {{5\.0+e-0*}}1], align 16
  // CHECK: @_ZN5test22t2E = dso_local global double* @_ZN5test21A1d
  // CHECK: @_ZN5test22t3E = dso_local global {{.*}} @_ZN5test21A1g
  double t0 = A::d;
  double t1[] = { A::d, A::f };
  const double *t2 = &a.d;
  int (*t3)() = &a.g;
}

// We don't expect to fold this in the frontend, but make sure it doesn't crash.
// CHECK: @PR9558 = dso_local global float 0.000000e+0
float PR9558 = reinterpret_cast<const float&>("asd");

// An initialized const automatic variable cannot be promoted to a constant
// global if it has a mutable member.
struct MutableMember {
  mutable int n;
};
int writeToMutable() {
  // CHECK-NOT: {{.*}}MM{{.*}} = {{.*}}constant
  const MutableMember MM = { 0 };
  return ++MM.n;
}

// Make sure we don't try to fold this in the frontend; the backend can't
// handle it.
// CHECK: @PR11705 = dso_local global i128 0
__int128_t PR11705 = (__int128_t)&PR11705;

// Make sure we don't try to fold this either.
// CHECK: @_ZZ23UnfoldableAddrLabelDiffvE1x = internal global i128 0
void UnfoldableAddrLabelDiff() { static __int128_t x = (long)&&a-(long)&&b; a:b:return;}

// But make sure we do fold this.
// CHECK: @_ZZ21FoldableAddrLabelDiffvE1x = internal global i64 sub (i64 ptrtoint (i8* blockaddress(@_Z21FoldableAddrLabelDiffv
void FoldableAddrLabelDiff() { static long x = (long)&&a-(long)&&b; a:b:return;}

// CHECK: @i = dso_local constant i32* bitcast (float* @PR9558 to i32*)
int &i = reinterpret_cast<int&>(PR9558);

int arr[2];
// CHECK: @pastEnd = dso_local constant i32* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i32]* @arr to i8*), i64 8) to i32*)
int &pastEnd = arr[2];

struct X {
  long n : 8;
};
long k;
X x = {(long)&k};
// CHECK: store i8 ptrtoint (i64* @k to i8), i8* getelementptr inbounds (%struct.X, %struct.X* @x, i32 0, i32 0)
