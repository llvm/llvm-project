// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

extern "C" int printf(...);

int init = 100;

struct M {
  int iM;
  M() : iM(init++) {}
};

struct N {
  int iN;
  N() : iN(200) {}
  N(N const & arg){this->iN = arg.iN; }
};

struct P {
  int iP;
  P() : iP(init++) {}
};


// CHECK-LABEL: define linkonce_odr void @_ZN1XC1ERKS_(%struct.X* {{[^,]*}} %this, %struct.X* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
struct X  : M, N, P { // ...
  X() : f1(1.0), d1(2.0), i1(3), name("HELLO"), bf1(0xff), bf2(0xabcd),
        au_i1(1234), au1_4("MASKED") {}
  P p0;
  void pr() {
    printf("iM = %d iN = %d, m1.iM = %d\n", iM, iN, m1.iM); 
    printf("im = %d p0.iP = %d, p1.iP = %d\n", iP, p0.iP, p1.iP); 
    printf("f1 = %f  d1 = %f  i1 = %d name(%s) \n", f1, d1, i1, name);
    printf("bf1 = %x  bf2 = %x\n", bf1, bf2);
    printf("au_i2 = %d\n", au_i2); 
    printf("au1_1 = %s\n", au1_1); 
  }
  M m1;
  P p1;
  float f1;
  double d1;
  int i1;
  const char *name;
  unsigned bf1 : 8;
  unsigned bf2 : 16;
  int arr[2];
  _Complex float complex;

  union {
    int au_i1;
    int au_i2;
  };
  union {
    const char * au1_1;
    float au1_2;
    int au1_3;
    const char * au1_4;
  };
};

static int ix = 1;
// class with user-defined copy constructor.
struct S {
  S() : iS(ix++) {  }
  S(const S& arg) { *this = arg; }
  int iS;
};

// class with trivial copy constructor.
struct I {
  I() : iI(ix++) {  }
  int iI;
};

struct XM {
  XM() {  }
  double dXM;
  S ARR_S[3][4][2];
  void pr() {
   for (unsigned i = 0; i < 3; i++)
     for (unsigned j = 0; j < 4; j++)
      for (unsigned k = 0; k < 2; k++)
        printf("ARR_S[%d][%d][%d] = %d\n", i,j,k, ARR_S[i][j][k].iS);
   for (unsigned i = 0; i < 3; i++)
      for (unsigned k = 0; k < 2; k++)
        printf("ARR_I[%d][%d] = %d\n", i,k, ARR_I[i][k].iI);
  }
  I ARR_I[3][2];
};

int main() {
  X a;
  X b(a);
  b.pr();
  X x;
  X c(x);
  c.pr();

  XM m0;
  XM m1 = m0;
  m1.pr();
}

struct A {
};

struct B : A {
  A &a;
};

void f(const B &b1) {
  B b2(b1);
}

// PR6628
namespace PR6628 {

struct T {
  T();
  ~T();

  double d;
};

struct A {
  A(const A &other, const T &t = T(), const T& t2 = T());
};

struct B : A {
  A a1;
  A a2;
  A a[10];
};

// Force the copy constructor to be synthesized.
void f(B b1) {
  B b2 = b1;
}

// CHECK:    define linkonce_odr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[A:%.*]]* @_ZN12rdar138169401AaSERKS0_(
// CHECK:      [[THIS:%.*]] = load [[A]]*, [[A]]**
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 1
// CHECK-NEXT: [[OTHER:%.*]] = load [[A]]*, [[A]]**
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OTHER]], i32 0, i32 1
// CHECK-NEXT: [[T4:%.*]] = bitcast i16* [[T0]] to i8*
// CHECK-NEXT: [[T5:%.*]] = bitcast i16* [[T2]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T4]], i8* align 8 [[T5]], i64 8, i1 false)
// CHECK-NEXT: ret [[A]]* [[THIS]]

// CHECK-LABEL: define linkonce_odr void @_ZN6PR66281BC2ERKS0_(%"struct.PR6628::B"* {{[^,]*}} %this, %"struct.PR6628::B"* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281AC2ERKS0_RKNS_1TES5_
// CHECK: call void @_ZN6PR66281TD1Ev
// CHECK: call void @_ZN6PR66281TD1Ev
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281AC1ERKS0_RKNS_1TES5_
// CHECK: call void @_ZN6PR66281TD1Ev
// CHECK: call void @_ZN6PR66281TD1Ev
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281TC1Ev
// CHECK: call void @_ZN6PR66281AC1ERKS0_RKNS_1TES5_
// CHECK: call void @_ZN6PR66281TD1Ev
// CHECK: call void @_ZN6PR66281TD1Ev

// CHECK-LABEL:    define linkonce_odr void @_ZN12rdar138169401AC2ERKS0_(
// CHECK:      [[THIS:%.*]] = load [[A]]*, [[A]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[THIS]] to i32 (...)***
// CHECK-NEXT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN12rdar138169401AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 1
// CHECK-NEXT: [[OTHER:%.*]] = load [[A]]*, [[A]]**
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OTHER]], i32 0, i32 1
// CHECK-NEXT: [[T4:%.*]] = bitcast i16* [[T0]] to i8*
// CHECK-NEXT: [[T5:%.*]] = bitcast i16* [[T2]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T4]], i8* align 8 [[T5]], i64 8, i1 false)
// CHECK-NEXT: ret void
}

// rdar://13816940
// Test above because things get weirdly re-ordered.
namespace rdar13816940 {
  struct A {
    virtual ~A();
    unsigned short a : 1;
    unsigned short : 15;
    unsigned other;
  };

  void test(A &a) {
    A x = a; // force copy constructor into existence
    x = a; // also force the copy assignment operator
  }
}
