// RUN: not %clang_cc1 -triple x86_64-apple-darwin -verify -emit-llvm -o - %s | FileCheck %s
void t1() {
  // CHECK-LABEL: define void @_Z2t1v
  // CHECK: [[REFLOAD:%.*]] = load i32*, i32** @a, align 8
  // CHECK: load i32, i32* [[REFLOAD]], align 4
  extern int& a;
  int b = a; 
}

void t2(int& a) {
  // CHECK-LABEL: define void @_Z2t2Ri
  // CHECK: [[REFLOAD2:%.*]] = load i32*, i32** {{.*}}, align 8
  // CHECK: load i32, i32* [[REFLOAD2]], align 4
  int b = a;
}

int g;
int& gr = g;
int& grr = gr;
void t3() {
  int b = gr;
}

// Test reference binding.

struct C { int a; };
void f(const bool&);
void f(const int&);
void f(const _Complex int&);
void f(const C&);

C aggregate_return();

bool& bool_reference_return();
int& int_reference_return();
_Complex int& complex_int_reference_return();
C& aggregate_reference_return();

void test_bool() {
  bool a = true;
  f(a);

  f(true);
  
  bool_reference_return() = true;
  a = bool_reference_return();
  
  struct { const bool& b; } b = { true };
}

void test_scalar() {
  int a = 10;
  f(a);
  
  struct { int bitfield : 3; } s = { 3 };
  f(s.bitfield);
  
  f(10);

  __attribute((vector_size(16))) typedef int vec4;
  f((vec4){1,2,3,4}[0]);
  
  int_reference_return() = 10;
  a = int_reference_return();
  
  struct { const int& a; } agg = { 10 };
}

void test_complex() {
  _Complex int a = 10i;
  f(a);
  
  f(10i);
  
  complex_int_reference_return() = 10i;
  a = complex_int_reference_return();
  
  struct { const _Complex int &a; } agg = { 10i };
}

void test_aggregate() {
  C c;
  f(c);

  f(aggregate_return());
  aggregate_reference_return().a = 10;

  c = aggregate_reference_return();
  
  struct { const C& a; } agg = { C() };
}

int& reference_return() {
  return g;
}

int reference_decl() {
  int& a = g;
  const int& b = 1;
  return a+b;
}

struct A {
  int& b();
};

void f(A* a) {
  int b = a->b();
}

// PR5122
void *foo = 0;
void * const & kFoo = foo;

struct D : C { D(); ~D(); };

void h() {
  // CHECK: call void @_ZN1DD1Ev
  const C& c = D();
}

namespace T {
  struct A {
    A();
    ~A();
  };

  struct B {
    B();
    ~B();
    A f();
  };

  void f() {
    // CHECK: call void @_ZN1T1BC1Ev
    // CHECK: call void @_ZN1T1B1fEv
    // CHECK: call void @_ZN1T1BD1Ev
    const A& a = B().f();
    // CHECK: call void @_ZN1T1fEv
    f();
    // CHECK: call void @_ZN1T1AD1Ev
  }
}

// PR5227.
namespace PR5227 {
void f(int &a) {
  (a = 10) = 20;
}
}

// PR5590
struct s0;
struct s1 { struct s0 &s0; };
void f0(s1 a) { s1 b = a; }

// PR6024
// CHECK: @_Z2f2v()
// CHECK: alloca i32,
// CHECK-NEXT: store
// CHECK-NEXT: ret
const int &f2() { return 0; }

// Don't constant fold const reference parameters with default arguments to
// their default arguments.
namespace N1 {
  const int foo = 1;
  // CHECK: @_ZN2N14test
  void test(const int& arg = foo) {
    // Ensure this array is on the stack where we can set values instead of
    // being a global constant.
    // CHECK: %args_array = alloca
    const int* const args_array[] = { &arg };
  }
}

// Bind to subobjects while extending the life of the complete object.
namespace N2 {
  class X {
  public:
    X(const X&);
    X &operator=(const X&);
    ~X();
  };

  struct P {
    X first;
  };

  P getP();

  // CHECK-LABEL: define void @_ZN2N21fEi
  // CHECK: call void @_ZN2N24getPEv
  // CHECK: getelementptr inbounds
  // CHECK: store i32 17
  // CHECK: call void @_ZN2N21PD1Ev
  void f(int i) {
    const X& xr = getP().first;
    i = 17;
  }

  struct SpaceWaster {
    int i, j;
  };

  struct ReallyHasX {
    X x;
  };

  struct HasX : ReallyHasX { };

  struct HasXContainer {
    HasX has;
  };

  struct Y : SpaceWaster, HasXContainer { };
  struct Z : SpaceWaster, Y { };

  Z getZ();

  // CHECK-LABEL: define void @_ZN2N21gEi
  // CHECK: call void @_ZN2N24getZEv
  // CHECK: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK: store i32 19
  // CHECK: call void @_ZN2N21ZD1Ev
  // CHECK: ret void
  void g(int i) {
    const X &xr = getZ().has.x;
    i = 19;    
  }
}

namespace N3 {

// PR7326

struct A {
  explicit A(int);
  ~A();
};

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call void @_ZN2N31AC1Ei(%"struct.N3::A"* {{[^,]*}} @_ZGRN2N35sA123E_, i32 123)
// CHECK: call i32 @__cxa_atexit
// CHECK: ret void
const A &sA123 = A(123);
}

namespace N4 {
  
struct A {
  A();
  ~A();
};

void f() {
  // CHECK-LABEL: define void @_ZN2N41fEv
  // CHECK: call void @_ZN2N41AC1Ev(%"struct.N4::A"* {{[^,]*}} @_ZGRZN2N41fEvE2ar_)
  // CHECK: call i32 @__cxa_atexit
  // CHECK: ret void
  static const A& ar = A();
  
}
}

// PR9494
namespace N5 {
struct AnyS { bool b; };
void f(const bool&);
AnyS g();
void h() {
  // CHECK: call i8 @_ZN2N51gEv()
  // CHECK: call void @_ZN2N51fERKb(i8*
  f(g().b);
}
}

// PR9565
namespace PR9565 {
  struct a { int a : 10, b : 10; };
  // CHECK-LABEL: define void @_ZN6PR95651fEv()
  void f() {
    // CHECK: call void @llvm.memcpy
    a x = { 0, 0 };
    // CHECK: [[WITH_SEVENTEEN:%[.a-zA-Z0-9]+]] = or i32 [[WITHOUT_SEVENTEEN:%[.a-zA-Z0-9]+]], 17
    // CHECK: store i32 [[WITH_SEVENTEEN]], i32* [[XA:%[.a-zA-Z0-9]+]]
    x.a = 17;
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: load
    // CHECK-NEXT: shl
    // CHECK-NEXT: ashr
    // CHECK-NEXT: store i32
    // CHECK-NEXT: store i32*
    const int &y = x.a;
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: load
    // CHECK-NEXT: and
    // CHECK-NEXT: or i32 {{.*}}, 19456
    // CHECK-NEXT: store i32
    x.b = 19;
    // CHECK-NEXT: ret void
  }
}

namespace N6 {
  extern struct x {char& x;}y;
  int a() { return y.x; }
  // CHECK-LABEL: define i32 @_ZN2N61aEv
  // CHECK: [[REFLOAD3:%.*]] = load i8*, i8** getelementptr inbounds (%"struct.N6::x", %"struct.N6::x"* @_ZN2N61yE, i32 0, i32 0), align 8
  // CHECK: load i8, i8* [[REFLOAD3]], align 1
}
