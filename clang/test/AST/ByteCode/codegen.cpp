// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fcxx-exceptions                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fcxx-exceptions -fexperimental-new-constant-interpreter | FileCheck %s

#ifdef __SIZEOF_INT128__
// CHECK: @PR11705 = global i128 0
__int128_t PR11705 = (__int128_t)&PR11705;
#endif

int arr[2];
// CHECK: @pastEnd = constant ptr getelementptr (i8, ptr @arr, i64 8)
int &pastEnd = arr[2];

// CHECK: @F = constant ptr @arr, align 8
int &F = arr[0];

// CHECK: @_ZL1q = internal global [4294967294 x i8] zeroinitializer, align 16
static char q[-2U];
void useQ() { char *p = q + 1; }

struct S {
  int a;
  float c[3];
};

// CHECK: @s = global %struct.S zeroinitializer, align 4
S s;
// CHECK: @sp = constant ptr getelementptr (i8, ptr @s, i64 16), align 8
float &sp = s.c[3];

// CHECK: @PR9558 = global float 0.000000e+0
float PR9558 = reinterpret_cast<const float&>("asd");
// CHECK: @i = constant ptr @PR9558
int &i = reinterpret_cast<int&>(PR9558);

namespace NearlyZeroInit {
  // CHECK: @_ZN14NearlyZeroInit1bE ={{.*}} global{{.*}} { i32, <{ i32, [2147483647 x i32] }> } { i32 1, <{ i32, [2147483647 x i32] }> <{ i32 2, [2147483647 x i32] zeroinitializer }> }{{.*}}
  struct B { int n; int arr[1024 * 1024 * 1024 * 2u]; } b = {1, {2}};
}

namespace BaseClassOffsets {
  struct A { int a; };
  struct B { int b; };
  struct C : A, B { int c; };

  extern C c;
  // CHECK: @_ZN16BaseClassOffsets1aE = global ptr @_ZN16BaseClassOffsets1cE, align 8
  A* a = &c;
  // CHECK: @_ZN16BaseClassOffsets1bE = global ptr getelementptr (i8, ptr @_ZN16BaseClassOffsets1cE, i64 4), align 8
  B* b = &c;
}

namespace ExprBase {
  struct A { int n; };
  struct B { int n; };
  struct C : A, B {};

  extern const int &&t = ((B&&)C{}).n;
  // CHECK: @_ZGRN8ExprBase1tE_ = internal global {{.*}} zeroinitializer,
  // CHECK: @_ZN8ExprBase1tE = constant ptr {{.*}} @_ZGRN8ExprBase1tE_, {{.*}} 8
}

namespace reinterpretcast {
  const unsigned int n = 1234;
  extern const int &s = reinterpret_cast<const int&>(n);
  // CHECK: @_ZN15reinterpretcastL1nE = internal constant i32 1234, align 4
  // CHECK: @_ZN15reinterpretcast1sE = constant ptr @_ZN15reinterpretcastL1nE, align 8

  void *f1(unsigned long l) {
    return reinterpret_cast<void *>(l);
  }
  // CHECK: define {{.*}} ptr @_ZN15reinterpretcast2f1Em
  // CHECK: inttoptr
}

namespace Bitfield {
  struct S { int a : 5; ~S(); };
  // CHECK: alloca
  // CHECK: call {{.*}}memset
  // CHECK: store i32 {{.*}}, ptr @_ZGRN8Bitfield1rE_
  // CHECK: call void @_ZN8Bitfield1SD1
  // CHECK: store ptr @_ZGRN8Bitfield1rE_, ptr @_ZN8Bitfield1rE, align 8
  int &&r = S().a;
}

namespace Null {
  decltype(nullptr) null();
  // CHECK: call {{.*}} @_ZN4Null4nullEv(
  int *p = null();
  struct S {};
  // CHECK: call {{.*}} @_ZN4Null4nullEv(
  int S::*q = null();
}

struct A {
  A();
  ~A();
  enum E { Foo };
};

A *g();

void f(A *a) {
  A::E e1 = a->Foo;

  // CHECK: call noundef ptr @_Z1gv()
  A::E e2 = g()->Foo;
  // CHECK: call void @_ZN1AC1Ev(
  // CHECK: call void @_ZN1AD1Ev(
  A::E e3 = A().Foo;
}

int notdead() {
  auto l = [c=0]() mutable {
    return  c++ < 5 ? 10 : 12;
  };
  return l();
}
// CHECK: _ZZ7notdeadvEN3$_0clEv
// CHECK: ret i32 %cond

/// The conmparison of those two parameters should NOT work.
bool paramcmp(const int& lhs, const int& rhs) {
  if (&lhs == &rhs)
    return true;
  return false;
}
// CHECK: _Z8paramcmpRKiS0_
// CHECK: if.then
// CHECK: if.end

/// &x == &OuterX should work and return 0.
class X {
public:
  X();
  X(const X&);
  X(const volatile X &);
  ~X();
};

extern X OuterX;

X test24() {
  X x;
  if (&x == &OuterX)
    throw 0;
  return x;
}
// CHECK: _Z6test24v
// CHECK-NOT: eh.resume
// CHECK-NOT: unreachable
