// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 %s -emit-llvm -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck -check-prefix=CHECK-PR10720 %s < %t

extern "C" int printf(...);

struct M {
  M() { printf("M()\n"); }
  M(int i) { iM = i; printf("M(%d)\n", i); }
  int iM;
  void MPR() {printf("iM = %d\n", iM); };
};

struct P {
  P() { printf("P()\n"); }
  P(int i) { iP = i; printf("P(%d)\n", i); }
  int iP;
  void PPR() {printf("iP = %d\n", iP); };
};

struct Q {
  Q() { printf("Q()\n"); }
  Q(int i) { iQ = i; printf("Q(%d)\n", i); }
  int iQ;
  void QPR() {printf("iQ = %d\n", iQ); };
};

struct N : M , P, Q {
  N() : f1(1.314), P(2000), ld(00.1234+f1), M(1000), Q(3000),
        d1(3.4567), i1(1234), m1(100) { printf("N()\n"); }
  M m1;
  M m2;
  float f1;
  int i1;
  float d1;
  void PR() {
    printf("f1 = %f d1 = %f i1 = %d ld = %f \n", f1,d1,i1, ld); 
    MPR();
    PPR();
    QPR();
    printf("iQ = %d\n", iQ);
    printf("iP = %d\n", iP);
    printf("iM = %d\n", iM);
    // FIXME. We don't yet support this syntax.
    // printf("iQ = %d\n", (*this).iQ);
    printf("iQ = %d\n", this->iQ);
    printf("iP = %d\n", this->iP);
    printf("iM = %d\n", this->iM);
  }
  float ld;
  float ff;
  M arr_m[3];
  P arr_p[1][3];
  Q arr_q[2][3][4];
};

int main() {
  M m1;

  N n1;
  n1.PR();
}

// PR5826
template <class T> struct A {
  A() {}
  A(int) {}
  A(const A&) {}
  ~A() {}
  operator int() {return 0;}
};

// CHECK-LABEL: define void @_Z1fv()
void f() {
  // CHECK: call void @_ZN1AIsEC1Ei
  A<short> a4 = 97;

  // CHECK-NEXT: store i32 17
  int i = 17;

  // CHECK-NEXT: call void @_ZN1AIsED1Ev
  // CHECK-NOT: call void @_ZN1AIsED1Ev
  // CHECK: ret void
}

// Make sure we initialize the vtable pointer if it's required by a
// base initializer.
namespace InitVTable {
  struct A { A(int); };
  struct B : A {
    virtual int foo();
    B();
    B(int);
  };

  // CHECK-LABEL: define void @_ZN10InitVTable1BC2Ev(%"struct.InitVTable::B"* %this) unnamed_addr
  // CHECK:      [[T0:%.*]] = bitcast [[B:%.*]]* [[THIS:%.*]] to i32 (...)***
  // CHECK-NEXT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN10InitVTable1BE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** [[T0]]
  // CHECK:      [[VTBL:%.*]] = load i32 ([[B]]*)**, i32 ([[B]]*)*** {{%.*}}
  // CHECK-NEXT: [[FNP:%.*]] = getelementptr inbounds i32 ([[B]]*)*, i32 ([[B]]*)** [[VTBL]], i64 0
  // CHECK-NEXT: [[FN:%.*]] = load i32 ([[B]]*)*, i32 ([[B]]*)** [[FNP]]
  // CHECK-NEXT: [[ARG:%.*]] = call i32 [[FN]]([[B]]* [[THIS]])
  // CHECK-NEXT: call void @_ZN10InitVTable1AC2Ei({{.*}}* {{%.*}}, i32 [[ARG]])
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[B]]* [[THIS]] to i32 (...)***
  // CHECK-NEXT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN10InitVTable1BE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** [[T0]]
  // CHECK-NEXT: ret void
  B::B() : A(foo()) {}

  // CHECK-LABEL: define void @_ZN10InitVTable1BC2Ei(%"struct.InitVTable::B"* %this, i32 %x) unnamed_addr
  // CHECK:      [[ARG:%.*]] = add nsw i32 {{%.*}}, 5
  // CHECK-NEXT: call void @_ZN10InitVTable1AC2Ei({{.*}}* {{%.*}}, i32 [[ARG]])
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[B]]* {{%.*}} to i32 (...)***
  // CHECK-NEXT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN10InitVTable1BE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** [[T0]]
  // CHECK-NEXT: ret void
  B::B(int x) : A(x + 5) {}
}

namespace rdar9694300 {
  struct X {
    int x;
  };

  // CHECK-LABEL: define void @_ZN11rdar96943001fEv
  void f() {
    // CHECK: alloca
    X x;
    // CHECK-NEXT: [[I:%.*]] = alloca i32
    // CHECK-NEXT: store i32 17, i32* [[I]]
    int i = 17;
    // CHECK-NEXT: ret void
  }
}

// Check that we emit a zero initialization step for list-value-initialization
// which calls a trivial default constructor.
namespace PR13273 {
  struct U {
    int t;
    U() = default;
  };

  struct S : U {
    S() = default;
  };

  // CHECK: define {{.*}}@_ZN7PR132731fEv(
  int f() {
    // CHECK-NOT: }
    // CHECK: llvm.memset{{.*}}i8 0
    return (new S{})->t;
  }
}

template<typename T>
struct X {
  X(const X &);

  T *start;
  T *end;
};

template<typename T> struct X;

// Make sure that the instantiated constructor initializes start and
// end properly.
// CHECK-LABEL: define linkonce_odr void @_ZN1XIiEC2ERKS0_(%struct.X* %this, %struct.X* dereferenceable({{[0-9]+}}) %other) unnamed_addr
// CHECK: {{store.*null}}
// CHECK: {{store.*null}}
// CHECK: ret
template<typename T>
X<T>::X(const X &other) : start(0), end(0) { }

X<int> get_X(X<int> x) { return x; }

namespace PR10720 {
  struct X { 
    X(const X&); 
    X(X&&); 
    X& operator=(const X&);
    X& operator=(X&&);
    ~X(); 
  };

  struct pair2 {
    X second[4];

    // CHECK-PR10720: define linkonce_odr {{.*}} @_ZN7PR107205pair2aSERKS0_
    // CHECK-PR10720: load
    // CHECK-PR10720: icmp ne
    // CHECK-PR10720-NEXT: br i1
    // CHECK-PR10720: call {{.*}} @_ZN7PR107201XaSERKS0_
    // CHECK-PR10720: ret
    pair2 &operator=(const pair2&) = default;

    // CHECK-PR10720: define linkonce_odr {{.*}} @_ZN7PR107205pair2aSEOS0_
    // CHECK-PR10720: load
    // CHECK-PR10720: icmp ne
    // CHECK-PR10720-NEXT: br i1
    // CHECK-PR10720: call {{.*}} @_ZN7PR107201XaSEOS0_
    // CHECK-PR10720: ret
    pair2 &operator=(pair2&&) = default;

    // CHECK-PR10720-LABEL: define linkonce_odr void @_ZN7PR107204pairC2ERKS0_
    // CHECK-PR10720-NOT: ret
    // CHECK-PR10720: call void @llvm.memcpy
    // CHECK-PR10720-NEXT: ret void

    // CHECK-PR10720-LABEL: define linkonce_odr void @_ZN7PR107205pair2C2ERKS0_
    // CHECK-PR10720-NOT: ret
    // CHECK-PR10720: call void @_ZN7PR107201XC1ERKS0_
    // CHECK-PR10720: icmp eq
    // CHECK-PR10720-NEXT: br i1
    // CHECK-PR10720: ret void

    // CHECK-PR10720-LABEL: define linkonce_odr void @_ZN7PR107205pair2C2EOS0_
    // CHECK-PR10720-NOT: ret
    // CHECK-PR10720: load
    // CHECK-PR10720: call void @_ZN7PR107201XC1EOS0_
    // CHECK-PR10720: icmp eq
    // CHECK-PR10720-NEXT: br i1
    // CHECK-PR10720: ret void
    pair2(pair2&&) = default;

    pair2(const pair2&) = default;
  };

  struct pair : X { // Make the copy constructor non-trivial, so we actually generate it.
    int second[4];
    pair(const pair&) = default;
  };

  void foo(const pair &x, const pair2 &x2) {
    pair y(x);
    pair2 y2(x2);
    pair2 y2m(static_cast<pair2&&>(y2));

    y2 = x2;
    y2m = static_cast<pair2&&>(y2);
  }

}
