// RUN: %clang_cc1 %s -emit-llvm -o %t.ll -triple=x86_64-apple-darwin10
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck -check-prefix=CHECK-GLOBAL %s < %t.ll
// RUN: %clang_cc1 %s -emit-llvm -o %t-opt.ll -triple=x86_64-apple-darwin10 -O3
// RUN: FileCheck --check-prefix=CHECK-O3 %s < %t-opt.ll

struct A { int a; int b; };
struct B { int b; };
struct C : B, A { };

// Zero init.
namespace ZeroInit {
  // CHECK-GLOBAL: @_ZN8ZeroInit1aE = global i64 -1
  int A::* a;
  
  // CHECK-GLOBAL: @_ZN8ZeroInit2aaE = global [2 x i64] [i64 -1, i64 -1]
  int A::* aa[2];
  
  // CHECK-GLOBAL: @_ZN8ZeroInit3aaaE = global [2 x [2 x i64]] {{\[}}[2 x i64] [i64 -1, i64 -1], [2 x i64] [i64 -1, i64 -1]]
  int A::* aaa[2][2];
  
  // CHECK-GLOBAL: @_ZN8ZeroInit1bE = global i64 -1,
  int A::* b = 0;

  // CHECK-GLOBAL: @_ZN8ZeroInit2saE = internal global %struct.anon { i64 -1 }
  struct {
    int A::*a;
  } sa;
  void test_sa() { (void) sa; } // force emission
  
  // CHECK-GLOBAL: @_ZN8ZeroInit3ssaE = internal
  // CHECK-GLOBAL: [2 x i64] [i64 -1, i64 -1]
  struct {
    int A::*aa[2];
  } ssa[2];
  void test_ssa() { (void) ssa; }
  
  // CHECK-GLOBAL: @_ZN8ZeroInit2ssE = internal global %struct.anon.1 { %struct.anon.2 { i64 -1 } }
  struct {
    struct {
      int A::*pa;
    } s;
  } ss;
  void test_ss() { (void) ss; }
  
  struct A {
    int A::*a;
    int b;
  };

  struct B {
    A a[10];
    char c;
    int B::*b;
  };

  struct C : A, B { int j; };
  // CHECK-GLOBAL: @_ZN8ZeroInit1cE = global {{%.*}} { %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::B" { [10 x %"struct.ZeroInit::A"] [%"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }, %"struct.ZeroInit::A" { i64 -1, i32 0 }], i8 0, i64 -1 }, i32 0 }, align 8
  C c;
}

// PR5674
namespace PR5674 {
  // CHECK-GLOBAL: @_ZN6PR56742pbE = global i64 4
  int A::*pb = &A::b;
}

// Casts.
namespace Casts {

int A::*pa;
int C::*pc;

void f() {
  // CHECK:      store i64 -1, i64* @_ZN5Casts2paE
  pa = 0;

  // CHECK-NEXT: [[TMP:%.*]] = load i64* @_ZN5Casts2paE, align 8
  // CHECK-NEXT: [[ADJ:%.*]] = add nsw i64 [[TMP]], 4
  // CHECK-NEXT: [[ISNULL:%.*]] = icmp eq i64 [[TMP]], -1
  // CHECK-NEXT: [[RES:%.*]] = select i1 [[ISNULL]], i64 [[TMP]], i64 [[ADJ]]
  // CHECK-NEXT: store i64 [[RES]], i64* @_ZN5Casts2pcE
  pc = pa;

  // CHECK-NEXT: [[TMP:%.*]] = load i64* @_ZN5Casts2pcE, align 8
  // CHECK-NEXT: [[ADJ:%.*]] = sub nsw i64 [[TMP]], 4
  // CHECK-NEXT: [[ISNULL:%.*]] = icmp eq i64 [[TMP]], -1
  // CHECK-NEXT: [[RES:%.*]] = select i1 [[ISNULL]], i64 [[TMP]], i64 [[ADJ]]
  // CHECK-NEXT: store i64 [[RES]], i64* @_ZN5Casts2paE
  pa = static_cast<int A::*>(pc);
}

}

// Comparisons
namespace Comparisons {
  void f() {
    int A::*a;

    // CHECK: icmp ne i64 {{.*}}, -1
    if (a) { }

    // CHECK: icmp ne i64 {{.*}}, -1
    if (a != 0) { }
    
    // CHECK: icmp ne i64 -1, {{.*}}
    if (0 != a) { }

    // CHECK: icmp eq i64 {{.*}}, -1
    if (a == 0) { }

    // CHECK: icmp eq i64 -1, {{.*}}
    if (0 == a) { }
  }
}

namespace ValueInit {

struct A {
  int A::*a;

  char c;

  A();
};

// CHECK: define void @_ZN9ValueInit1AC2Ev(%"struct.ValueInit::A"* %this) unnamed_addr
// CHECK: store i64 -1, i64*
// CHECK: ret void
A::A() : a() {}

}

namespace PR7139 {

struct pair {
  int first;
  int second;
};

typedef int pair::*ptr_to_member_type;

struct ptr_to_member_struct { 
  ptr_to_member_type data;
  int i;
};

struct A {
  ptr_to_member_struct a;

  A() : a() {}
};

// CHECK-O3: define zeroext i1 @_ZN6PR71395checkEv() nounwind readnone
bool check() {
  // CHECK-O3: ret i1 true
  return A().a.data == 0;
}

// CHECK-O3: define zeroext i1 @_ZN6PR71396check2Ev() nounwind readnone
bool check2() {
  // CHECK-O3: ret i1 true
  return ptr_to_member_type() == 0;
}

}

namespace VirtualBases {

struct A {
  char c;
  int A::*i;
};

// CHECK-GLOBAL: @_ZN12VirtualBases1bE = global %"struct.VirtualBases::B" { i32 (...)** null, %"struct.VirtualBases::A" { i8 0, i64 -1 } }, align 8
struct B : virtual A { };
B b;

// CHECK-GLOBAL: @_ZN12VirtualBases1cE = global %"struct.VirtualBases::C" { i32 (...)** null, i64 -1, %"struct.VirtualBases::A" { i8 0, i64 -1 } }, align 8
struct C : virtual A { int A::*i; };
C c;

// CHECK-GLOBAL: @_ZN12VirtualBases1dE = global %"struct.VirtualBases::D" { %"struct.VirtualBases::C.base" { i32 (...)** null, i64 -1 }, i64 -1, %"struct.VirtualBases::A" { i8 0, i64 -1 } }, align 8
struct D : C { int A::*i; };
D d;

}

namespace Test1 {

// Don't crash when A contains a bit-field.
struct A {
  int A::* a;
  int b : 10;
};
A a;

}

namespace BoolPtrToMember {
  struct X {
    bool member;
  };

  // CHECK: define i8* @_ZN15BoolPtrToMember1fERNS_1XEMS0_b
  bool &f(X &x, bool X::*member) {
    // CHECK: {{bitcast.* to i8\*}}
    // CHECK-NEXT: getelementptr inbounds i8*
    // CHECK-NEXT: ret i8*
    return x.*member;
  }
}

namespace PR8507 {
  
struct S;
void f(S* p, double S::*pm) {
  if (0 < p->*pm) {
  }
}

}

namespace test4 {
  struct A             { int A_i; };
  struct B : virtual A { int A::*B_p; };
  struct C : virtual B { int    *C_p; };
  struct D :         C { int    *D_p; };

  // CHECK-GLOBAL: @_ZN5test41dE = global %"struct.test4::D" { %"struct.test4::C.base" zeroinitializer, i32* null, %"struct.test4::B.base" { i32 (...)** null, i64 -1 }, %"struct.test4::A" zeroinitializer }, align 8
  D d;
}

namespace PR11487 {
  union U
  {
    int U::* mptr;
    char x[16];
  } x;
  // CHECK-GLOBAL: @_ZN7PR114871xE = global %"union.PR11487::U" { i64 -1, [8 x i8] zeroinitializer }, align 8
  
}

namespace PR13097 {
  struct X { int x; X(const X&); };
  struct A {
    int qq;
      X x;
  };
  A f();
  X g() { return f().*&A::x; }
  // CHECK: define void @_ZN7PR130971gEv
  // CHECK: call void @_ZN7PR130971fEv
  // CHECK-NOT: memcpy
  // CHECK: call void @_ZN7PR130971XC1ERKS0_
}
