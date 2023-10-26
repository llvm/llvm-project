// RUN: %clang_cc1 -w -fmerge-all-constants -triple x86_64-elf-gnu -emit-llvm -o - %s -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -w -fmerge-all-constants -triple x86_64-elf-gnu -emit-llvm -o - %s -std=c++20 | FileCheck -check-prefix=CHECK20 %s

// FIXME: The padding in all these objects should be zero-initialized.
namespace StructUnion {
  struct A {
    int n;
    double d;
    union U {
      constexpr U(int x) : x(x) {}
      constexpr U(const char *y) : y(y) {}
      int x;
      const char *y;
    } u;

    constexpr A(int n, double d, int x) : n(n), d(d), u(x) {}
    constexpr A(int n, double d, const char *y) : n(n), d(d), u(y) {}
  };

  // CHECK: @_ZN11StructUnion1aE ={{.*}} constant {{.*}} { i32 1, double 2.000000e+00, {{.*}} { i32 3, [4 x i8] undef } }
  extern constexpr A a(1, 2.0, 3);

  // CHECK: @_ZN11StructUnion1bE ={{.*}} constant {{.*}} { i32 4, double 5.000000e+00, {{.*}} { ptr @{{.*}} } }
  extern constexpr A b(4, 5, "hello");

  struct B {
    int n;
  };

  // CHECK: @_ZN11StructUnion1cE ={{.*}} global {{.*}} zeroinitializer
  // CHECK: @_ZN11StructUnion2c2E ={{.*}} global {{.*}} zeroinitializer
  B c;
  B c2 = B();

  // CHECK: @_ZN11StructUnion1dE ={{.*}} global {{.*}} zeroinitializer
  B d[10];

  struct C {
    constexpr C() : c(0) {}
    int c;
  };

  // CHECK: @_ZN11StructUnion1eE ={{.*}} global {{.*}} zeroinitializer
  C e[10];

  struct D {
    constexpr D() : d(5) {}
    int d;
  };

  // CHECK: @_ZN11StructUnion1fE ={{.*}} global {{.*}} { i32 5 }
  D f;

  union E {
    int a;
    void *b = &f;
  };

  // CHECK: @_ZN11StructUnion1gE ={{.*}} global {{.*}} @_ZN11StructUnion1fE
  E g;

  // CHECK: @_ZN11StructUnion1hE ={{.*}} global {{.*}} @_ZN11StructUnion1fE
  E h = E();
}

namespace BaseClass {
  template<typename T, unsigned> struct X : T {};
  struct C { char c = 1; };
  template<unsigned... Ns> struct Cs : X<C,Ns>... {};
  struct N { int n = 3; };
  struct D { double d = 4.0; };

  template<typename ...Ts>
  struct Test : Ts... { constexpr Test() : Ts()..., n(5) {} int n; };

  using Test1 = Test<N, C, Cs<1,2>, D, X<C,1>>;
  // CHECK: @_ZN9BaseClass2t1E ={{.*}} constant {{.*}} { i32 3, i8 1, i8 1, i8 1, double 4.000000e+00, i8 1, i32 5 }, align 8
  extern constexpr Test1 t1 = Test1();

  struct DN : D, N {};
  struct DND : DN, X<D,0> {};
  struct DNN : DN, X<N,0> {};
  // CHECK: @_ZN9BaseClass3dndE ={{.*}} constant {{.*}} { double 4.000000e+00, i32 3, double 4.000000e+00 }
  extern constexpr DND dnd = DND();
  // Note, N subobject is laid out in DN subobject's tail padding.
  // CHECK: @_ZN9BaseClass3dnnE ={{.*}} constant {{.*}} { double 4.000000e+00, i32 3, i32 3 }
  extern constexpr DNN dnn = DNN();

  struct E {};
  struct Test2 : X<E,0>, X<E,1>, X<E,2>, X<E,3> {};
  // CHECK: @_ZN9BaseClass2t2E ={{.*}} constant {{.*}} zeroinitializer, align 1
  extern constexpr Test2 t2 = Test2();

  struct __attribute((packed)) PackedD { double y = 2; };
  struct Test3 : C, PackedD { constexpr Test3() {} };
  // CHECK: @_ZN9BaseClass2t3E ={{.*}} constant <{ i8, double }> <{ i8 1, double 2.000000e+00 }>
  extern constexpr Test3 t3 = Test3();
}

namespace Array {
  // CHECK: @_ZN5Array3arrE ={{.*}} constant [2 x i32] [i32 4, i32 0]
  extern constexpr int arr[2] = { 4 };

  // CHECK: @_ZN5Array1cE ={{.*}} constant [6 x [4 x i8]] [{{.*}} c"foo\00", [4 x i8] c"a\00\00\00", [4 x i8] c"bar\00", [4 x i8] c"xyz\00", [4 x i8] c"b\00\00\00", [4 x i8] c"123\00"]
  extern constexpr char c[6][4] = { "foo", "a", { "bar" }, { 'x', 'y', 'z' }, { "b" }, '1', '2', '3' };

  // CHECK: @_ZN5Array2ucE ={{.*}} constant [4 x i8] c"foo\00"
  extern constexpr unsigned char uc[] = { "foo" };

  struct C { constexpr C() : n(5) {} int n, m = 3 * n + 1; };
  // CHECK: @_ZN5Array5ctorsE ={{.*}} constant [3 x {{.*}}] [{{.*}} { i32 5, i32 16 }, {{.*}} { i32 5, i32 16 }, {{.*}} { i32 5, i32 16 }]
  extern const C ctors[3];
  constexpr C ctors[3];

  // CHECK: @_ZN5Array1dE ={{.*}} constant {{.*}} { [2 x i32] [i32 1, i32 2], [3 x i32] [i32 3, i32 4, i32 5] }
  struct D { int n[2]; int m[3]; } extern constexpr d = { 1, 2, 3, 4, 5 };

  struct E {
    char c[4];
    char d[4];
    constexpr E() : c("foo"), d("x") {}
  };
  // CHECK: @_ZN5Array1eE ={{.*}} constant {{.*}} { [4 x i8] c"foo\00", [4 x i8] c"x\00\00\00" }
  extern constexpr E e = E();

  // PR13290
  struct F { constexpr F() : n(4) {} int n; };
  // CHECK: @_ZN5Array2f1E ={{.*}} global {{.*}} zeroinitializer
  F f1[1][1][0] = { };
  // CHECK: @_ZN5Array2f2E ={{.*}} global {{.* i32 4 .* i32 4 .* i32 4 .* i32 4 .* i32 4 .* i32 4 .* i32 4 .* i32 4}}
  F f2[2][2][2] = { };
}

namespace MemberPtr {
  struct B1 {
    int a, b;
    virtual void f();
    void g();
  };
  struct B2 {
    int c, d;
    virtual void h();
    void i();
  };
  struct C : B1 {
    int e;
    virtual void j();
    void k();
  };
  struct D : C, B2 {
    int z;
    virtual void l();
    void m();
  };

  // CHECK: @_ZN9MemberPtr2daE ={{.*}} constant i64 8
  // CHECK: @_ZN9MemberPtr2dbE ={{.*}} constant i64 12
  // CHECK: @_ZN9MemberPtr2dcE ={{.*}} constant i64 32
  // CHECK: @_ZN9MemberPtr2ddE ={{.*}} constant i64 36
  // CHECK: @_ZN9MemberPtr2deE ={{.*}} constant i64 16
  // CHECK: @_ZN9MemberPtr2dzE ={{.*}} constant i64 40
  extern constexpr int (D::*da) = &B1::a;
  extern constexpr int (D::*db) = &C::b;
  extern constexpr int (D::*dc) = &B2::c;
  extern constexpr int (D::*dd) = &D::d;
  extern constexpr int (D::*de) = &C::e;
  extern constexpr int (D::*dz) = &D::z;

  // CHECK: @_ZN9MemberPtr2baE ={{.*}} constant i64 8
  // CHECK: @_ZN9MemberPtr2bbE ={{.*}} constant i64 12
  // CHECK: @_ZN9MemberPtr2bcE ={{.*}} constant i64 8
  // CHECK: @_ZN9MemberPtr2bdE ={{.*}} constant i64 12
  // CHECK: @_ZN9MemberPtr2beE ={{.*}} constant i64 16
  // CHECK: @_ZN9MemberPtr3b1zE ={{.*}} constant i64 40
  // CHECK: @_ZN9MemberPtr3b2zE ={{.*}} constant i64 16
  extern constexpr int (B1::*ba) = (int(B1::*))&B1::a;
  extern constexpr int (B1::*bb) = (int(B1::*))&C::b;
  extern constexpr int (B2::*bc) = (int(B2::*))&B2::c;
  extern constexpr int (B2::*bd) = (int(B2::*))&D::d;
  extern constexpr int (B1::*be) = (int(B1::*))&C::e;
  extern constexpr int (B1::*b1z) = (int(B1::*))&D::z;
  extern constexpr int (B2::*b2z) = (int(B2::*))&D::z;

  // CHECK: @_ZN9MemberPtr2dfE ={{.*}} constant {{.*}} { i64 1, i64 0 }
  // CHECK: @_ZN9MemberPtr2dgE ={{.*}} constant {{.*}} { i64 {{.*}}2B11gEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr2dhE ={{.*}} constant {{.*}} { i64 1, i64 24 }
  // CHECK: @_ZN9MemberPtr2diE ={{.*}} constant {{.*}} { i64 {{.*}}2B21iEv{{.*}}, i64 24 }
  // CHECK: @_ZN9MemberPtr2djE ={{.*}} constant {{.*}} { i64 9, i64 0 }
  // CHECK: @_ZN9MemberPtr2dkE ={{.*}} constant {{.*}} { i64 {{.*}}1C1kEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr2dlE ={{.*}} constant {{.*}} { i64 17, i64 0 }
  // CHECK: @_ZN9MemberPtr2dmE ={{.*}} constant {{.*}} { i64 {{.*}}1D1mEv{{.*}}, i64 0 }
  extern constexpr void (D::*df)() = &C::f;
  extern constexpr void (D::*dg)() = &B1::g;
  extern constexpr void (D::*dh)() = &B2::h;
  extern constexpr void (D::*di)() = &D::i;
  extern constexpr void (D::*dj)() = &C::j;
  extern constexpr void (D::*dk)() = &C::k;
  extern constexpr void (D::*dl)() = &D::l;
  extern constexpr void (D::*dm)() = &D::m;

  // CHECK: @_ZN9MemberPtr2bfE ={{.*}} constant {{.*}} { i64 1, i64 0 }
  // CHECK: @_ZN9MemberPtr2bgE ={{.*}} constant {{.*}} { i64 {{.*}}2B11gEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr2bhE ={{.*}} constant {{.*}} { i64 1, i64 0 }
  // CHECK: @_ZN9MemberPtr2biE ={{.*}} constant {{.*}} { i64 {{.*}}2B21iEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr2bjE ={{.*}} constant {{.*}} { i64 9, i64 0 }
  // CHECK: @_ZN9MemberPtr2bkE ={{.*}} constant {{.*}} { i64 {{.*}}1C1kEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr3b1lE ={{.*}} constant {{.*}} { i64 17, i64 0 }
  // CHECK: @_ZN9MemberPtr3b1mE ={{.*}} constant {{.*}} { i64 {{.*}}1D1mEv{{.*}}, i64 0 }
  // CHECK: @_ZN9MemberPtr3b2lE ={{.*}} constant {{.*}} { i64 17, i64 -24 }
  // CHECK: @_ZN9MemberPtr3b2mE ={{.*}} constant {{.*}} { i64 {{.*}}1D1mEv{{.*}}, i64 -24 }
  extern constexpr void (B1::*bf)()  = (void(B1::*)())&C::f;
  extern constexpr void (B1::*bg)()  = (void(B1::*)())&B1::g;
  extern constexpr void (B2::*bh)()  = (void(B2::*)())&B2::h;
  extern constexpr void (B2::*bi)()  = (void(B2::*)())&D::i;
  extern constexpr void (B1::*bj)()  = (void(B1::*)())&C::j;
  extern constexpr void (B1::*bk)()  = (void(B1::*)())&C::k;
  extern constexpr void (B1::*b1l)() = (void(B1::*)())&D::l;
  extern constexpr void (B1::*b1m)() = (void(B1::*)())&D::m;
  extern constexpr void (B2::*b2l)() = (void(B2::*)())&D::l;
  extern constexpr void (B2::*b2m)() = (void(B2::*)())&D::m;
}

namespace LiteralReference {
  struct Lit {
    constexpr Lit() : n(5) {}
    int n;
  };

  // This creates a non-const temporary and binds a reference to it.
  // CHECK: @[[TEMP:.*]] = internal global {{.*}} { i32 5 }, align 4
  // CHECK: @_ZN16LiteralReference3litE ={{.*}} constant {{.*}} @[[TEMP]], align 8
  const Lit &lit = Lit();

  // This creates a const temporary as part of the reference initialization.
  // CHECK: @[[TEMP:.*]] = internal constant {{.*}} { i32 5 }, align 4
  // CHECK: @_ZN16LiteralReference4lit2E ={{.*}} constant {{.*}} @[[TEMP]], align 8
  const Lit &lit2 = {};

  struct A { int &&r1; const int &&r2; };
  struct B { A &&a1; const A &&a2; };
  B b = { { 0, 1 }, { 2, 3 } };
  // CHECK: @[[TEMP0:.*]] = internal global i32 0, align 4
  // CHECK: @[[TEMP1:.*]] = internal constant i32 1, align 4
  // CHECK: @[[TEMPA1:.*]] = internal global {{.*}} { ptr @[[TEMP0]], ptr @[[TEMP1]] }, align 8
  // CHECK: @[[TEMP2:.*]] = internal global i32 2, align 4
  // CHECK: @[[TEMP3:.*]] = internal constant i32 3, align 4
  // CHECK: @[[TEMPA2:.*]] = internal constant {{.*}} { ptr @[[TEMP2]], ptr @[[TEMP3]] }, align 8
  // CHECK: @_ZN16LiteralReference1bE ={{.*}} global {{.*}} { ptr @[[TEMPA1]], ptr @[[TEMPA2]] }, align 8

  struct Subobj {
    int a, b, c;
  };
  // CHECK: @[[TEMP:.*]] = internal global {{.*}} { i32 1, i32 2, i32 3 }, align 4
  // CHECK: @_ZN16LiteralReference2soE ={{.*}} constant ptr getelementptr (i8, ptr @[[TEMP]], i64 4)
  constexpr int &&so = Subobj{ 1, 2, 3 }.b;

  struct Dummy { int padding; };
  struct Derived : Dummy, Subobj {
    constexpr Derived() : Dummy{200}, Subobj{4, 5, 6} {}
  };
  using ConstDerived = const Derived;
  // CHECK: @[[TEMPCOMMA:.*]] = internal constant {{.*}} { i32 200, i32 4, i32 5, i32 6 }
  // CHECK: @_ZN16LiteralReference5commaE ={{.*}} constant {{.*}} getelementptr {{.*}} @[[TEMPCOMMA]]{{.*}}, i64 8)
  constexpr const int &comma = (1, (2, ConstDerived{}).b);

  // CHECK: @[[TEMPDERIVED:.*]] = internal global {{.*}} { i32 200, i32 4, i32 5, i32 6 }
  // CHECK: @_ZN16LiteralReference4baseE ={{.*}} constant {{.*}} getelementptr {{.*}} @[[TEMPDERIVED]]{{.*}}, i64 4)
  constexpr Subobj &&base = Derived{};

  // CHECK: @_ZN16LiteralReference7derivedE ={{.*}} constant {{.*}} @[[TEMPDERIVED]]
  constexpr Derived &derived = static_cast<Derived&>(base);
}

namespace NonLiteralConstexpr {
  constexpr int factorial(int n) {
    return n ? factorial(n-1) * n : 1;
  }
  extern void f(int *p);

  struct NonTrivialDtor {
    constexpr NonTrivialDtor() : n(factorial(5)), p(&n) {}
    ~NonTrivialDtor() {
      f(p);
    }

    int n;
    int *p;
  };
  static_assert(!__is_literal(NonTrivialDtor), "");
  // CHECK: @_ZN19NonLiteralConstexpr3ntdE ={{.*}} global {{.*}} { i32 120, ptr @
  NonTrivialDtor ntd;

  struct VolatileMember {
    constexpr VolatileMember() : n(5) {}
    volatile int n;
  };
  static_assert(!__is_literal(VolatileMember), "");
  // CHECK: @_ZN19NonLiteralConstexpr2vmE ={{.*}} global {{.*}} { i32 5 }
  VolatileMember vm;

  struct Both {
    constexpr Both() : n(10) {}
    ~Both();
    volatile int n;
  };
  // CHECK: @_ZN19NonLiteralConstexpr1bE ={{.*}} global {{.*}} { i32 10 }
  Both b;

  void StaticVars() {
    // CHECK: @_ZZN19NonLiteralConstexpr10StaticVarsEvE3ntd = {{.*}} { i32 120, ptr @
    // CHECK: @_ZGVZN19NonLiteralConstexpr10StaticVarsEvE3ntd =
    static NonTrivialDtor ntd;
    // CHECK: @_ZZN19NonLiteralConstexpr10StaticVarsEvE2vm = {{.*}} { i32 5 }
    // CHECK-NOT: @_ZGVZN19NonLiteralConstexpr10StaticVarsEvE2vm =
    static VolatileMember vm;
    // CHECK: @_ZZN19NonLiteralConstexpr10StaticVarsEvE1b = {{.*}} { i32 10 }
    // CHECK: @_ZGVZN19NonLiteralConstexpr10StaticVarsEvE1b =
    static Both b;
  }
}

// PR12067
namespace VirtualMembers {
  struct A {
    constexpr A(double d) : d(d) {}
    virtual void f();
    double d;
  };
  struct B : A {
    constexpr B() : A(2.0), c{'h', 'e', 'l', 'l', 'o'} {}
    constexpr B(int n) : A(n), c{'w', 'o', 'r', 'l', 'd'} {}
    virtual void g();
    char c[5];
  };
  struct C {
    constexpr C() : n(64) {}
    int n;
  };
  struct D : C, A, B {
    constexpr D() : A(1.0), B(), s(5) {}
    short s;
  };
  struct E : D, B {
    constexpr E() : B(3), c{'b','y','e'} {}
    char c[3];
  };
  // CHECK: @_ZN14VirtualMembers1eE ={{.*}} global { ptr, double, i32, ptr, double, [5 x i8], i16, ptr, double, [5 x i8], [3 x i8] } { ptr getelementptr inbounds ({ [3 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTVN14VirtualMembers1EE, i32 0, inrange i32 0, i32 2), double 1.000000e+00, i32 64, ptr getelementptr inbounds ({ [3 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTVN14VirtualMembers1EE, i32 0, inrange i32 1, i32 2), double 2.000000e+00, [5 x i8] c"hello", i16 5, ptr getelementptr inbounds ({ [3 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTVN14VirtualMembers1EE, i32 0, inrange i32 2, i32 2), double 3.000000e+00, [5 x i8] c"world", [3 x i8] c"bye" }
  E e;

  struct nsMemoryImpl {
    virtual void f();
  };
  // CHECK: @_ZN14VirtualMembersL13sGlobalMemoryE = internal global %"struct.VirtualMembers::nsMemoryImpl" { ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN14VirtualMembers12nsMemoryImplE, i32 0, inrange i32 0, i32 2) }
  __attribute__((used))
  static nsMemoryImpl sGlobalMemory;

  template<class T>
  struct TemplateClass {
    constexpr TemplateClass() : t{42} {}
    virtual void templateMethod() {}

    T t;
  };
  // CHECK: @_ZN14VirtualMembers1tE ={{.*}} global { ptr, i32 } { ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN14VirtualMembers13TemplateClassIiEE, i32 0, inrange i32 0, i32 2), i32 42 }
  TemplateClass<int> t;
}

namespace PR13273 {
  struct U {
    int t;
    U() = default;
  };

  struct S : U {
    S() = default;
  };

  // CHECK: @_ZN7PR132731sE = {{.*}} zeroinitializer
  extern const S s {};
}

namespace ArrayTemporary {
  struct A { const int (&x)[3]; };
  struct B { const A (&x)[2]; };
  // CHECK: @[[A1:_ZGRN14ArrayTemporary1bE.*]] = internal constant [3 x i32] [i32 1, i32 2, i32 3]
  // CHECK: @[[A2:_ZGRN14ArrayTemporary1bE.*]] = internal constant [3 x i32] [i32 4, i32 5, i32 6]
  // CHECK: @[[ARR:_ZGRN14ArrayTemporary1bE.*]] = internal constant [2 x {{.*}}] [{{.*}} { ptr @[[A1]] }, {{.*}} { ptr @[[A2]] }]
  // CHECK: @[[B:_ZGRN14ArrayTemporary1bE.*]] = internal global {{.*}} { ptr @[[ARR]] }
  // CHECK: @_ZN14ArrayTemporary1bE ={{.*}} constant ptr @[[B]]
  B &&b = { { { { 1, 2, 3 } }, { { 4, 5, 6 } } } };
}

namespace UnemittedTemporaryDecl {
  constexpr int &&ref = 0;
  extern constexpr int &ref2 = ref;
  // CHECK: @_ZGRN22UnemittedTemporaryDecl3refE_ = internal global i32 0

  // FIXME: This declaration should not be emitted -- it isn't odr-used.
  // CHECK: @_ZN22UnemittedTemporaryDecl3refE

  // CHECK: @_ZN22UnemittedTemporaryDecl4ref2E ={{.*}} constant ptr @_ZGRN22UnemittedTemporaryDecl3refE_
}

namespace DR2126 {
  struct A { int &&b; };
  constexpr const A &a = {42};
  // CHECK: @_ZGRN6DR21261aE0_ = internal global i32 42
  // FIXME: This is unused and need not be emitted.
  // CHECK: @_ZGRN6DR21261aE_ = internal constant {{.*}} { ptr @_ZGRN6DR21261aE0_ }
  // CHECK: @_ZN6DR21261rE ={{.*}} constant ptr @_ZGRN6DR21261aE0_
  int &r = a.b;

  // Dynamically initialized: the temporary object bound to 'b' could be
  // modified (eg, by placement 'new') before the initializer of 's' runs.
  constexpr A &&b = {42};
  // CHECK: @_ZN6DR21261sE ={{.*}} global ptr null
  int &s = b.b;
}

// CHECK: @_ZZN12LocalVarInit3aggEvE1a = internal constant {{.*}} i32 101
// CHECK: @_ZZN12LocalVarInit4ctorEvE1a = internal constant {{.*}} i32 102
// CHECK: @__const._ZN12LocalVarInit8mutable_Ev.a = private unnamed_addr constant {{.*}} i32 103
// CHECK: @_ZGRN33ClassTemplateWithStaticDataMember1SIvE1aE_ = linkonce_odr constant i32 5, comdat
// CHECK: @_ZN33ClassTemplateWithStaticDataMember3useE ={{.*}} constant ptr @_ZGRN33ClassTemplateWithStaticDataMember1SIvE1aE_
// CHECK: @_ZGRN39ClassTemplateWithHiddenStaticDataMember1SIvE1aE_ = linkonce_odr hidden constant i32 5, comdat
// CHECK: @_ZN39ClassTemplateWithHiddenStaticDataMember3useE ={{.*}} constant ptr @_ZGRN39ClassTemplateWithHiddenStaticDataMember1SIvE1aE_
// CHECK: @.str.[[STR:[0-9]+]] ={{.*}} constant [9 x i8] c"12345678\00"
// CHECK-NEXT: @e = global %struct.PR69979 { ptr @.str.[[STR]] }
// CHECK: @_ZGRZN20InlineStaticConstRef3funEvE1i_ = linkonce_odr constant i32 10, comdat
// CHECK20: @_ZZN12LocalVarInit4dtorEvE1a = internal constant {{.*}} i32 103

// Constant initialization tests go before this point,
// dynamic initialization tests go after.

// We must emit a constant initializer for NonLiteralConstexpr::ntd, but also
// emit an initializer to register its destructor.
// CHECK: define {{.*}}cxx_global_var_init{{.*}}
// CHECK-NOT: NonLiteralConstexpr
// CHECK: call {{.*}}cxa_atexit(ptr @_ZN19NonLiteralConstexpr14NonTrivialDtorD1Ev, ptr @_ZN19NonLiteralConstexpr3ntdE
// CHECK-NEXT: ret void

// We don't need to emit any dynamic initialization for NonLiteralConstexpr::vm.
// CHECK-NOT: NonLiteralConstexpr2vm

// We must emit a constant initializer for NonLiteralConstexpr::b, but also
// emit an initializer to register its destructor.
// CHECK: define {{.*}}cxx_global_var_init{{.*}}
// CHECK-NOT: NonLiteralConstexpr
// CHECK: call {{.*}}cxa_atexit(ptr @_ZN19NonLiteralConstexpr4BothD1Ev, ptr @_ZN19NonLiteralConstexpr1bE
// CHECK-NEXT: ret void

// CHECK: define {{.*}}NonLiteralConstexpr10StaticVars
// CHECK-NOT: }
// CHECK: call {{.*}}cxa_atexit{{.*}}@_ZN19NonLiteralConstexpr14NonTrivialDtorD1Ev
// CHECK-NOT: }
// CHECK: call {{.*}}cxa_atexit{{.*}}@_ZN19NonLiteralConstexpr4BothD1Ev

// PR12848: Don't emit dynamic initializers for local constexpr variables.
namespace LocalVarInit {
  constexpr int f(int n) { return n; }
  struct Agg { int k; };
  struct Ctor { constexpr Ctor(int n) : k(n) {} int k; };
  struct Mutable { constexpr Mutable(int n) : k(n) {} mutable int k; };

  // CHECK: define {{.*}} @_ZN12LocalVarInit6scalarEv
  // CHECK-NOT: call
  // CHECK: store i32 100,
  // CHECK-NOT: call
  // CHECK: ret i32 100
  int scalar() { constexpr int a = { f(100) }; return a; }

  // CHECK: define {{.*}} @_ZN12LocalVarInit3aggEv
  // CHECK-NOT: call
  // CHECK: ret i32 101
  int agg() { constexpr Agg a = { f(101) }; return a.k; }

  // CHECK: define {{.*}} @_ZN12LocalVarInit4ctorEv
  // CHECK-NOT: call
  // CHECK: ret i32 102
  int ctor() { constexpr Ctor a = { f(102) }; return a.k; }

  // CHECK: define {{.*}} @_ZN12LocalVarInit8mutable_Ev
  // CHECK-NOT: call
  // CHECK: call {{.*}}memcpy{{.*}} @__const._ZN12LocalVarInit8mutable_Ev.a
  // CHECK-NOT: call
  // Can't fold return value due to 'mutable'.
  // CHECK-NOT: ret i32 103
  // CHECK: }
  int mutable_() { constexpr Mutable a = { f(103) }; return a.k; }

#if __cplusplus >= 202002L
  // CHECK20: define {{.*}} @_ZN12LocalVarInit4dtorEv
  // CHECK20-NOT: call
  // CHECK20: ret i32 103
  struct Dtor { constexpr Dtor(int n) : k(n) {} constexpr ~Dtor() {} int k; };
  int dtor() { constexpr Dtor a = { f(103) }; return a.k; }
#endif
}

namespace CrossFuncLabelDiff {
  // Make sure we refuse to constant-fold the variable b.
  constexpr long a(bool x) { return x ? 0 : (long)&&lbl + (0 && ({lbl: 0;})); }
  void test() { static long b = (long)&&lbl - a(false); lbl: return; }
  // CHECK: sub nsw i64 ptrtoint (ptr blockaddress(@_ZN18CrossFuncLabelDiff4testEv, {{.*}}) to i64),
  // CHECK: store i64 {{.*}}, ptr @_ZZN18CrossFuncLabelDiff4testEvE1b, align 8
}

// PR12012
namespace VirtualBase {
  struct B {};
  struct D : virtual B {};
  D d;
  // CHECK: call {{.*}}@_ZN11VirtualBase1DC1Ev

  template<typename T> struct X : T {
    constexpr X() : T() {}
  };
  X<D> x;
  // CHECK: call {{.*}}@_ZN11VirtualBase1XINS_1DEEC1Ev
}

// PR12145
namespace Unreferenced {
  int n;
  constexpr int *p = &n;
  // We must not emit a load of 'p' here, since it's not odr-used.
  int q = *p;
  // CHECK-NOT: _ZN12Unreferenced1pE
  // CHECK: = load i32, ptr @_ZN12Unreferenced1nE
  // CHECK-NEXT: store i32 {{.*}}, ptr @_ZN12Unreferenced1qE
  // CHECK-NOT: _ZN12Unreferenced1pE

  // Technically, we are not required to substitute variables of reference types
  // initialized by constant expressions, because the special case for odr-use
  // of variables in [basic.def.odr]p2 only applies to objects. But we do so
  // anyway.

  constexpr int &r = n;
  // CHECK-NOT: _ZN12Unreferenced1rE
  int s = r;

  const int t = 1;
  const int &rt = t;
  int f(int);
  int u = f(rt);
  // CHECK: call noundef i32 @_ZN12Unreferenced1fEi(i32 noundef 1)
}

namespace InitFromConst {
  template<typename T> void consume(T);

  const bool b = true;
  const int n = 5;
  constexpr double d = 4.3;

  struct S { int n = 7; S *p = 0; };
  constexpr S s = S();
  const S &r = s;
  constexpr const S *p = &r;
  constexpr int S::*mp = &S::n;
  constexpr int a[3] = { 1, 4, 9 };

  void test() {
    // CHECK: call void @_ZN13InitFromConst7consumeIbEEvT_(i1 noundef zeroext true)
    consume(b);

    // CHECK: call void @_ZN13InitFromConst7consumeIiEEvT_(i32 noundef 5)
    consume(n);

    // CHECK: call void @_ZN13InitFromConst7consumeIdEEvT_(double noundef 4.300000e+00)
    consume(d);

    // CHECK: call void @_ZN13InitFromConst7consumeIRKNS_1SEEEvT_(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @_ZN13InitFromConstL1sE)
    consume<const S&>(s);

    // CHECK: call void @_ZN13InitFromConst7consumeIRKNS_1SEEEvT_(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @_ZN13InitFromConstL1sE)
    consume<const S&>(r);

    // CHECK: call void @_ZN13InitFromConst7consumeIPKNS_1SEEEvT_(ptr noundef @_ZN13InitFromConstL1sE)
    consume(p);

    // CHECK: call void @_ZN13InitFromConst7consumeIMNS_1SEiEEvT_(i64 0)
    consume(mp);

    // CHECK: call void @_ZN13InitFromConst7consumeIPKiEEvT_(ptr noundef @_ZN13InitFromConstL1aE)
    consume(a);
  }
}

namespace Null {
  decltype(nullptr) null();
  // CHECK: call {{.*}} @_ZN4Null4nullEv(
  int *p = null();
  struct S {};
  // CHECK: call {{.*}} @_ZN4Null4nullEv(
  int S::*q = null();
}

namespace InlineStaticConstRef {
  inline const int &fun() {
    static const int &i = 10;
    return i;
    // CHECK: ret ptr @_ZGRZN20InlineStaticConstRef3funEvE1i_
  }
  const int &use = fun();
}

namespace ClassTemplateWithStaticDataMember {
  template <typename T>
  struct S {
    static const int &a;
  };
  template <typename T>
  const int &S<T>::a = 5;
  const int &use = S<void>::a;
}

namespace ClassTemplateWithHiddenStaticDataMember {
  template <typename T>
  struct S {
    __attribute__((visibility("hidden"))) static const int &a;
  };
  template <typename T>
  const int &S<T>::a = 5;
  const int &use = S<void>::a;
}

namespace ClassWithStaticConstexprDataMember {
struct X {
  static constexpr const char &p = 'c';
};

// CHECK: @_ZGRN34ClassWithStaticConstexprDataMember1X1pE_
const char *f() { return &X::p; }
}

struct PR69979 {
  const char (&d)[9];
} e {"12345678"};

// VirtualMembers::TemplateClass::templateMethod() must be defined in this TU,
// not just declared.
// CHECK: define linkonce_odr void @_ZN14VirtualMembers13TemplateClassIiE14templateMethodEv(ptr {{[^,]*}} %this)
