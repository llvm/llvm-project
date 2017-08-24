// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

template < bool condition, typename T = void >
struct enable_if { typedef T type; };

template< typename T >
struct enable_if< false, T > {};

// PR5876
namespace Casts {
  template< unsigned O >
  void implicit(typename enable_if< O <= 4 >::type* = 0) {
  }
  
  template< unsigned O >
  void cstyle(typename enable_if< O <= (unsigned)4 >::type* = 0) {
  }

  template< unsigned O >
  void functional(typename enable_if< O <= unsigned(4) >::type* = 0) {
  }
  
  template< unsigned O >
  void static_(typename enable_if< O <= static_cast<unsigned>(4) >::type* = 0) {
  }

  template <unsigned O, typename T>
  void reinterpret_(typename enable_if<O <= sizeof(reinterpret_cast<T *>(0))>::type * = 0) {
  }

  template <typename T, T *p>
  void const_(typename enable_if<0 <= sizeof(const_cast<T *>(p))>::type * = 0) {
  }

  template <typename T, T *p>
  void dynamic_(typename enable_if<0 <= sizeof(dynamic_cast<T *>(p))>::type * = 0) {
  }

  template< typename T >
  void auto_(decltype(new auto(T()))) {
  }

  template< typename T >
  void scalar_(decltype(T(), int())) {
  }

  template <unsigned N> struct T {};

  template <int N> T<N> f() { return T<N>(); }

  extern int i;
  extern struct S {} s;
  
  // CHECK-LABEL: define weak_odr void @_ZN5Casts8implicitILj4EEEvPN9enable_ifIXleT_Li4EEvE4typeE
  template void implicit<4>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts6cstyleILj4EEEvPN9enable_ifIXleT_cvjLi4EEvE4typeE
  template void cstyle<4>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts10functionalILj4EEEvPN9enable_ifIXleT_cvjLi4EEvE4typeE
  template void functional<4>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts7static_ILj4EEEvPN9enable_ifIXleT_scjLi4EEvE4typeE
  template void static_<4>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts12reinterpret_ILj4EiEEvPN9enable_ifIXleT_szrcPT0_Li0EEvE4typeE
  template void reinterpret_<4, int>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts6const_IiXadL_ZNS_1iEEEEEvPN9enable_ifIXleLi0EszccPT_T0_EvE4typeE
  template void const_<int, &i>(void*);
  // CHECK-LABEL: define weak_odr void @_ZN5Casts8dynamic_INS_1SEXadL_ZNS_1sEEEEEvPN9enable_ifIXleLi0EszdcPT_T0_EvE4typeE
  template void dynamic_<struct S, &s>(void*);

  // CHECK-LABEL: define weak_odr void @_ZN5Casts1fILi6EEENS_1TIXT_EEEv
  template T<6> f<6>();

  // CHECK-LABEL: define weak_odr void @_ZN5Casts5auto_IiEEvDTnw_DapicvT__EEE(
  template void auto_<int>(int*);

  // CHECK-LABEL: define weak_odr void @_ZN5Casts7scalar_IiEEvDTcmcvT__Ecvi_EE(
  template void scalar_<int>(int);
}

namespace test1 {
  short foo(short);
  int foo(int);

  // CHECK-LABEL: define linkonce_odr signext i16 @_ZN5test11aIsEEDTcl3foocvT__EEES1_(
  template <class T> auto a(T t) -> decltype(foo(T())) { return foo(t); }

  // CHECK-LABEL: define linkonce_odr signext i16 @_ZN5test11bIsEEDTcp3foocvT__EEES1_(
  template <class T> auto b(T t) -> decltype((foo)(T())) { return (foo)(t); }

  void test(short s) {
    a(s);
    b(s);
  }
}

namespace test2 {
  template <class T> void a(T x, decltype(x()) y) {}
  template <class T> auto b(T x) -> decltype(x()) { return x(); }
  template <class T> void c(T x, void (*p)(decltype(x()))) {}
  template <class T> void d(T x, auto (*p)() -> decltype(x())) {}
  template <class T> void e(auto (*p)(T y) -> decltype(y())) {}
  template <class T> void f(void (*p)(T x, decltype(x()) y)) {}
  template <class T> void g(T x, decltype(x()) y) {
    static decltype(x()) variable;
    variable = 0;
  }
  template <class T> void h(T x, decltype((decltype(x())(*)()) 0) y) {}
  template <class T> void i(decltype((auto (*)(T x) -> decltype(x())) 0) y) {}

  float foo();
  void bar(float);
  float baz(float(*)());
  void fred(float(*)(), float);

  // CHECK-LABEL: define void @_ZN5test211instantiateEv
  void instantiate() {
    // CHECK: call void @_ZN5test21aIPFfvEEEvT_DTclfL0p_EE(
    a(foo, 0.0f);
    // CHECK: call float @_ZN5test21bIPFfvEEEDTclfp_EET_(
    (void) b(foo);
    // CHECK: call void @_ZN5test21cIPFfvEEEvT_PFvDTclfL1p_EEE(
    c(foo, bar);
    // CHECK: call void @_ZN5test21dIPFfvEEEvT_PFDTclfL0p_EEvE(
    d(foo, foo);
    // CHECK: call void @_ZN5test21eIPFfvEEEvPFDTclfp_EET_E(
    e(baz);
    // CHECK: call void @_ZN5test21fIPFfvEEEvPFvT_DTclfL0p_EEE(
    f(fred);
    // CHECK: call void @_ZN5test21gIPFfvEEEvT_DTclfL0p_EE(
    g(foo, 0.0f);
    // CHECK: call void @_ZN5test21hIPFfvEEEvT_DTcvPFDTclfL0p_EEvELi0EE(
    h(foo, foo);
    // CHECK: call void @_ZN5test21iIPFfvEEEvDTcvPFDTclfp_EET_ELi0EE(
    i<float(*)()>(baz);
  }

  // CHECK: store float {{.*}}, float* @_ZZN5test21gIPFfvEEEvT_DTclfL0p_EEE8variable,
}

namespace test3 {
  template <class T, class U> void a(T x, U y, decltype(x.*y) z) {}  

  struct X {
    int *member;
  };

  // CHECK-LABEL: define void @_ZN5test311instantiateEv
  void instantiate() {
    X x;
    int *ip;
    // CHECK: call void @_ZN5test31aINS_1XEMS1_PiEEvT_T0_DTdsfL0p_fL0p0_E
    a(x, &X::member, ip);
  }
}

namespace test4 {
  struct X {
    X(int);
  };

  template <typename T>
  void tf1(decltype(new T(1)) p)
  {}

  template <typename T>
  void tf2(decltype(new T({1})) p)
  {}

  template <typename T>
  void tf3(decltype(new T{1}) p)
  {}

  // CHECK: void @_ZN5test43tf1INS_1XEEEvDTnw_T_piLi1EEE
  template void tf1<X>(X*);

  // CHECK: void @_ZN5test43tf2INS_1XEEEvDTnw_T_piilLi1EEEE
  template void tf2<X>(X*);

  // CHECK: void @_ZN5test43tf3INS_1XEEEvDTnw_T_ilLi1EEE
  template void tf3<X>(X*);

}

namespace test5 {
  template <typename T> void a(decltype(noexcept(T()))) {}
  template void a<int>(decltype(noexcept(int())));
  // CHECK: void @_ZN5test51aIiEEvDTnxcvT__EE(
}

namespace test6 {
  struct X {
    int i;
  };

  struct Y {
    union {
      int i;
    };
  };

  struct Z {
    union {
      X ua;
      Y ub;
    };

    struct {
      X s;
    };

    union {
      union {
        struct {
          struct {
            X uuss;
          };
        };
      };
    };
  };

  Z z, *zp;

  template<typename T>
  void f1(decltype(T(z.ua.i))) {}
  template void f1<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f1IiEEvDTcvT_dtdtL_ZNS_1zEE2ua1iE

  template<typename T>
  void f2(decltype(T(z.ub.i))) {}
  template void f2<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f2IiEEvDTcvT_dtdtL_ZNS_1zEE2ub1iE

  template<typename T>
  void f3(decltype(T(z.s.i))) {}
  template void f3<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f3IiEEvDTcvT_dtdtL_ZNS_1zEE1s1iE

  template<typename T>
  void f4(decltype(T(z.uuss.i))) {}
  template void f4<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f4IiEEvDTcvT_dtdtL_ZNS_1zEE4uuss1iE

  template<typename T>
  void f5(decltype(T(zp->ua.i))) {}
  template void f5<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f5IiEEvDTcvT_dtptL_ZNS_2zpEE2ua1iE

  template<typename T>
  void f6(decltype(T(zp->ub.i))) {}
  template void f6<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f6IiEEvDTcvT_dtptL_ZNS_2zpEE2ub1iE

  template<typename T>
  void f7(decltype(T(zp->s.i))) {}
  template void f7<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f7IiEEvDTcvT_dtptL_ZNS_2zpEE1s1iE

  template<typename T>
  void f8(decltype(T(zp->uuss.i))) {}
  template void f8<int>(int);
  // CHECK-LABEL: define weak_odr void @_ZN5test62f8IiEEvDTcvT_dtptL_ZNS_2zpEE4uuss1iE
}

namespace test7 {
  struct A { int x[3]; };
  struct B { B(int, int); } extern b;
  struct C { C(B); };
  struct D { D(C); };
  struct E { E(std::initializer_list<int>); };
  struct F { F(E); };

  template<class T> decltype(A{1,2},T()) fA1(T t) {}
  template<class T> decltype(A({1,2}),T()) fA2(T t) {}
  template<class T> decltype(B{1,2},T()) fB1(T t) {}
  template<class T> decltype(B({1,2}),T()) fB2(T t) {}
  template<class T> decltype(C{{1,2}},T()) fC1(T t) {}
  template<class T> decltype(C({1,2}),T()) fC2(T t) {}
  template<class T> decltype(D{b},T()) fD1(T t) {}
  template<class T> decltype(D(b),T()) fD2(T t) {}
  template<class T> decltype(E{1,2},T()) fE1(T t) {}
  template<class T> decltype(E({1,2}),T()) fE2(T t) {}
  template<class T> decltype(F{{1,2}},T()) fF1(T t) {}
  template<class T> decltype(F({1,2}),T()) fF2(T t) {}

  template<class T> decltype(T{}) fT1(T t) {}
  template<class T> decltype(T()) fT2(T t) {}
  template<class T> decltype(T{1}) fT3(T t) {}
  template<class T> decltype(T(1)) fT4(T t) {}
  template<class T> decltype(T{1,2}) fT5(T t) {}
  template<class T> decltype(T(1,2)) fT6(T t) {}
  template<class T> decltype(T{{}}) fT7(T t) {}
  template<class T> decltype(T({})) fT8(T t) {}
  template<class T> decltype(T{{1}}) fT9(T t) {}
  template<class T> decltype(T({1})) fTA(T t) {}
  template<class T> decltype(T{{1,2}}) fTB(T t) {}
  template<class T> decltype(T({1,2})) fTC(T t) {}

  int main() {
    fA1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fA1IiEEDTcmtlNS_1AELi1ELi2EEcvT__EES2_
    fA2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fA2IiEEDTcmcvNS_1AEilLi1ELi2EEcvT__EES2_
    fB1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fB1IiEEDTcmtlNS_1BELi1ELi2EEcvT__EES2_
    fB2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fB2IiEEDTcmcvNS_1BEilLi1ELi2EEcvT__EES2_
    fC1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fC1IiEEDTcmtlNS_1CEilLi1ELi2EEEcvT__EES2_
    fC2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fC2IiEEDTcmcvNS_1CEilLi1ELi2EEcvT__EES2_
    fD1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fD1IiEEDTcmtlNS_1DEL_ZNS_1bEEEcvT__EES2_
    fD2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fD2IiEEDTcmcvNS_1DEL_ZNS_1bEEcvT__EES2_
    fE1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fE1IiEEDTcmtlNS_1EELi1ELi2EEcvT__EES2_
    fE2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fE2IiEEDTcmcvNS_1EEilLi1ELi2EEcvT__EES2_
    fF1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fF1IiEEDTcmtlNS_1FEilLi1ELi2EEEcvT__EES2_
    fF2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fF2IiEEDTcmcvNS_1FEilLi1ELi2EEcvT__EES2_
    fT1(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fT1IiEEDTtlT_EES1_(
    fT2(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fT2IiEEDTcvT__EES1_(
    fT3(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fT3IiEEDTtlT_Li1EEES1_(
    fT4(1); // CHECK-LABEL: define {{.*}} @_ZN5test73fT4IiEEDTcvT_Li1EES1_(
    fT5(b); // CHECK-LABEL: define {{.*}} @_ZN5test73fT5INS_1BEEEDTtlT_Li1ELi2EEES2_(
    fT6(b); // CHECK-LABEL: define {{.*}} @_ZN5test73fT6INS_1BEEEDTcvT__Li1ELi2EEES2_(
    fT7(A{}); // CHECK-LABEL: define {{.*}} @_ZN5test73fT7INS_1AEEEDTtlT_ilEEES2_(
    fT8(A{}); // CHECK-LABEL: define {{.*}} @_ZN5test73fT8INS_1AEEEDTcvT_ilEES2_(
    fT9(A{}); // CHECK-LABEL: define {{.*}} @_ZN5test73fT9INS_1AEEEDTtlT_ilLi1EEEES2_(
    fTA(A{}); // CHECK-LABEL: define {{.*}} @_ZN5test73fTAINS_1AEEEDTcvT_ilLi1EEES2_(
    fTB<C>(b); // CHECK-LABEL: define {{.*}} @_ZN5test73fTBINS_1CEEEDTtlT_ilLi1ELi2EEEES2_(
    fTC<C>(b); // CHECK-LABEL: define {{.*}} @_ZN5test73fTCINS_1CEEEDTcvT_ilLi1ELi2EEES2_(
  }
}


namespace test8 {
  template <class>
  struct X {
    template<typename T> T foo() const { return 0; }
    template <class T> auto bar() const -> decltype(foo<T>()) { return 0; }
  };

  // CHECK-LABEL: define weak_odr i32 @_ZNK5test81XIiE3barIiEEDTcl3fooIT_EEEv
  template int X<int>::bar<int>() const;
}

namespace designated_init {
  struct A { struct B { int b[5][5]; } a; };
  // CHECK-LABEL: define {{.*}} @_ZN15designated_init1fINS_1AEEEvDTtlT_di1adi1bdxLi3EdXLi1ELi4ELi9EEE(
  template<typename T> void f(decltype(T{.a.b[3][1 ... 4] = 9}) x) {}
  void use_f(A a) { f<A>(a); }
}
