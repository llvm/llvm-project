// RUN: %clang_cc1 -verify -Wno-return-type -Wno-main -std=c++11 -fclang-abi-compat=latest -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// RUN: %clang_cc1 -verify -Wno-return-type -Wno-main -std=c++20 -fclang-abi-compat=latest -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s --check-prefixes=CHECK,CXX20
// expected-no-diagnostics

namespace test1 {
int x;
template <int& D> class T { };
// CHECK: void @_ZN5test12f0ENS_1TIL_ZNS_1xEEEE(
void f0(T<x> a0) {}
}

namespace test1 {
// CHECK: void @_ZN5test12f0Ef
void f0(float) {}
template<void (&)(float)> struct t1 {};
// CHECK: void @_ZN5test12f1ENS_2t1IL_ZNS_2f0EfEEE(
void f1(t1<f0> a0) {}
}

namespace test2 {
// CHECK: void @_ZN5test22f0Ef
void f0(float) {}
template<void (*)(float)> struct t1 {};
// CHECK: void @_ZN5test22f1ENS_2t1IXadL_ZNS_2f0EfEEEE(
void f1(t1<f0> a0) {}
}

namespace test3 {
// CHECK: void @test3_f0
extern "C" void test3_f0(float) {}
template<void (&)(float)> struct t1 {};
// CHECK: void @_ZN5test32f1ENS_2t1IL_Z8test3_f0EEE(
void f1(t1<test3_f0> a0) {}
}

namespace test4 {
// CHECK: void @test4_f0
extern "C" void test4_f0(float) {}
template<void (*)(float)> struct t1 {};
// CHECK: void @_ZN5test42f1ENS_2t1IXadL_Z8test4_f0EEEE(
void f1(t1<test4_f0> a0) {}
}

// CHECK: void @test5_f0
extern "C" void test5_f0(float) {}
int main(int) {}

namespace test5 {
template<void (&)(float)> struct t1 {};
// CHECK: void @_ZN5test52f1ENS_2t1IL_Z8test5_f0EEE(
void f1(t1<test5_f0> a0) {}

template<int (&)(int)> struct t2 {};
// CHECK: void @_ZN5test52f2ENS_2t2IL_Z4mainEEE
void f2(t2<main> a0) {}
}

namespace test6 {
struct A { void im0(float); };
// CHECK: void @_ZN5test61A3im0Ef
void A::im0(float) {}
template <void(A::*)(float)> class T { };
// CHECK: void @_ZN5test62f0ENS_1TIXadL_ZNS_1A3im0EfEEEE(
void f0(T<&A::im0> a0) {}
}

namespace test7 {
  template<typename T>
  struct meta {
    static const unsigned value = sizeof(T);
  };

  template<unsigned> struct int_c { 
    typedef float type;
  };

  template<typename T>
  struct X {
    template<typename U>
    X(U*, typename int_c<(meta<T>::value + meta<U>::value)>::type *) { }
  };

  // CHECK: define weak_odr {{.*}} @_ZN5test71XIiEC1IdEEPT_PNS_5int_cIXplL_ZNS_4metaIiE5valueEEsr4metaIS3_EE5valueEE4typeE(
  template X<int>::X(double*, float*);
}

namespace test8 {
  template<typename T>
  struct meta {
    struct type {
      static const unsigned value = sizeof(T);
    };
  };

  template<unsigned> struct int_c { 
    typedef float type;
  };

  template<typename T>
  void f(int_c<meta<T>::type::value>) { }

  // CHECK-LABEL: define weak_odr {{.*}}void @_ZN5test81fIiEEvNS_5int_cIXsr4metaIT_E4typeE5valueEEE(
  template void f<int>(int_c<sizeof(int)>);
}

namespace test9 {
  template<typename T>
  struct supermeta {
    template<typename U>
    struct apply {
      typedef T U::*type;
    };
  };

  struct X { };

  template<typename T, typename U>
  typename supermeta<T>::template apply<U>::type f();

  void test_f() {
    // CHECK: @_ZN5test91fIiNS_1XEEENS_9supermetaIT_E5applyIT0_E4typeEv()
    // Note: GCC incorrectly mangles this as
    // _ZN5test91fIiNS_1XEEENS_9supermetaIT_E5apply4typeEv, while EDG
    // gets it right.
    f<int, X>();
  }
}

namespace test10 {
  template<typename T>
  struct X {
    template<typename U>
    struct definition {
    };
  };

  // CHECK: _ZN6test101fIidEENS_1XIT_E10definitionIT0_EES2_S5_
  template<typename T, typename U>
  typename X<T>::template definition<U> f(T, U) { }

  void g(int i, double d) {
    f(i, d);
  }
}

// Report from cxx-abi-dev, 2012.01.04.
namespace test11 {
  int cmp(char a, char b);
  template <typename T, int (*cmp)(T, T)> struct A {};
  template <typename T> void f(A<T,cmp> &) {}
  template void f<char>(A<char,cmp> &);
  // CHECK: @_ZN6test111fIcEEvRNS_1AIT_L_ZNS_3cmpEccEEE(
}

namespace test12 {
  // Make sure we can mangle non-type template args with internal linkage.
  static int f() {}
  const int n = 10;
  template<typename T, T v> void test() {}
  void use() {
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIFivETnT_XadL_ZNS_L1fEvEEEEvv(
    test<int(), &f>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIRFivETnT_L_ZNS_L1fEvEEEvv(
    test<int(&)(), f>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIPKiTnT_XadL_ZNS_L1nEEEEEvv(
    test<const int*, &n>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIRKiTnT_L_ZNS_L1nEEEEvv(
    test<const int&, n>();
  }
}

// Test the boundary condition of minimal signed integers.
namespace test13 {
  template <char c> char returnChar() { return c; }
  template char returnChar<-128>();
  // CHECK: @_ZN6test1310returnCharILcn128EEEcv()

  template <short s> short returnShort() { return s; }
  template short returnShort<-32768>();
  // CHECK: @_ZN6test1311returnShortILsn32768EEEsv()
}

namespace test14 {
  template <typename> inline int inl(bool b) {
    if (b) {
      static struct {
        int field;
      } a;
      // CHECK: @_ZZN6test143inlIvEEibE1a

      return a.field;
    } else {
      static struct {
        int field;
      } a;
      // CHECK: @_ZZN6test143inlIvEEibE1a_0

      return a.field;
    }
  }

  int call(bool b) { return inl<void>(b); }
}

namespace std {
template <class _Tp, _Tp...> struct integer_sequence {};
}

namespace test15 {
template <int N>
__make_integer_seq<std::integer_sequence, int, N> make() {}
template __make_integer_seq<std::integer_sequence, int, 5> make<5>();
// CHECK: define weak_odr {{.*}} @_ZN6test154makeILi5EEE18__make_integer_seqISt16integer_sequenceiXT_EEv(
}

namespace test16 {
  // Ensure we properly form substitutions for template names in prefixes.
  // CHECK: @_ZN6test161fINS_1TEEEvNT_1UIiE1VIiEENS5_IfEE
  template<typename T> void f(typename T::template U<int>::template V<int>, typename T::template U<int>::template V<float>);
  struct T { template<typename I> struct U { template<typename J> using V = int; }; };
  void g() { f<T>(1, 2); }
}

#if __cplusplus >= 202002L
namespace cxx20 {
  template<auto> struct A {};
  template<typename T, T V> struct B {};
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AILf3f800000EEE(
  void f(A<1.0f>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AILd3ff0000000000000EEE(
  void f(A<1.0>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AILe3fff8000000000000000EEE(
  void f(A<1.0l>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXtlCiLi0ELi1EEEEE(
  void f(A<1i>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXtlCdLd0000000000000000ELd3ff0000000000000EEEEE(
  void f(A<1.0i>) {}

  int x;
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXadL_ZNS_1xEEEEE(
  void f(A<&x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPiXadL_ZNS_1xEEEEE(
  void f(B<int*, &x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvPKiadL_ZNS_1xEEEEE(
  void f(A<(const int*)&x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPKiXadL_ZNS_1xEEEEE(
  void f(B<const int*, &x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvPvadL_ZNS_1xEEEEE(
  void f(A<(void*)&x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPvXadL_ZNS_1xEEEEE(
  void f(B<void*, (void*)&x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvPKvadL_ZNS_1xEEEEE(
  void f(A<(const void*)&x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPKvXadL_ZNS_1xEEEEE(
  void f(B<const void*, (const void*)&x>) {}

  struct Q { int x; } q;

  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXadsoiL_ZNS_1qEEEEEE(
  void f(A<&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPiXadsoiL_ZNS_1qEEEEEE(
  void f(B<int*, &q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXadsoKiL_ZNS_1qEEEEEE(
  void f(A<(const int*)&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPKiXadsoS1_L_ZNS_1qEEEEEE
  void f(B<const int*, (const int*)&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvPvadsoiL_ZNS_1qEEEEEE(
  void f(A<(void*)&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPvXadsoiL_ZNS_1qEEEEEE(
  void f(B<void*, (void*)&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvPKvadsoiL_ZNS_1qEEEEEE(
  void f(A<(const void*)&q.x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIPKvXadsoiL_ZNS_1qEEEEEE(
  void f(B<const void*, (const void*)&q.x>) {}

  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXadL_ZNS_1Q1xEEEEE(
  void f(A<&Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIMNS_1QEiXadL_ZNS1_1xEEEEE
  void f(B<int Q::*, &Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXcvMNS_1QEKiadL_ZNS1_1xEEEEE(
  void f(A<(const int Q::*)&Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIMNS_1QEKiXadL_ZNS1_1xEEEEE(
  void f(B<const int Q::*, (const int Q::*)&Q::x>) {}

  struct R : Q {};

  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXmcMNS_1REiadL_ZNS_1Q1xEEEEEE(
  void f(A<(int R::*)&Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIMNS_1REiXmcS2_adL_ZNS_1Q1xEEEEEE(
  void f(B<int R::*, (int R::*)&Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1AIXmcMNS_1REKiadL_ZNS_1Q1xEEEEEE(
  void f(A<(const int R::*)&Q::x>) {}
  // CXX20: define {{.*}} @_ZN5cxx201fENS_1BIMNS_1REKiXmcS3_adL_ZNS_1Q1xEEEEEE(
  void f(B<const int R::*, (const int R::*)&Q::x>) {}
}
#endif

namespace test17 {
  // Ensure we mangle the types for non-type template arguments if we've lost
  // track of argument / parameter correspondence.
  template<int A, int ...B> struct X {};

  // CHECK: define {{.*}} @_ZN6test171fILi1EJLi2ELi3ELi4EEEEvNS_1XIXT_EJLi5EXspT0_ELi6EEEE
  template<int D, int ...C> void f(X<D, 5u, C..., 6u>) {}
  void g() { f<1, 2, 3, 4>({}); }

  // Note: there is no J...E here, because we can't form a pack argument, and
  // the 5u and 6u are mangled with the original type 'j' (unsigned int) not
  // with the resolved type 'i' (signed int).
  // CHECK: define {{.*}} @_ZN6test171hILi4EJLi1ELi2ELi3EEEEvNS_1XIXspT0_ELj5EXT_ELj6EEE
  template<int D, int ...C> void h(X<C..., 5u, D, 6u>) {}
  void i() { h<4, 1, 2, 3>({}); }

#if __cplusplus >= 201402L
  template<int A, const volatile int*> struct Y {};
  int n;
  // Case 1: &n is a resolved template argument, with a known parameter:
  // mangled with no conversion.
  // CXX20: define {{.*}} @_ZN6test172j1ILi1EEEvNS_1YIXT_EXadL_ZNS_1nEEEEE
  template<int N> void j1(Y<N, (const int*)&n>) {}
  // Case 2: &n is an unresolved template argument, with an unknown
  // corresopnding parameter: mangled as the source expression.
  // CXX20: define {{.*}} @_ZN6test172j2IJLi1EEEEvNS_1YIXspT_EXcvPKiadL_ZNS_1nEEEEE
  template<int ...Ns> void j2(Y<Ns..., (const int*)&n>) {}
  // Case 3: &n is a resolved template argument, with a known parameter, but
  // for a template that can be overloaded on type: mangled with the parameter type.
  // CXX20: define {{.*}} @_ZN6test172j3ILi1EEEvDTplT_clL_ZNS_1yIXcvPVKiadL_ZNS_1nEEEEEivEEE
  template<const volatile int*> int y();
  template<int N> void j3(decltype(N + y<(const int*)&n>())) {}
  void k() {
    j1<1>(Y<1, &n>());
    j2<1>(Y<1, &n>());
    j3<1>(0);
  }
#endif
}

namespace partially_dependent_template_args {
  namespace test1 {
    template<bool B> struct enable { using type = int; };
    template<typename ...> struct and_ { static constexpr bool value = true; };
    template<typename T> inline typename enable<and_<T, T, T>::value>::type f(T) {}
    // FIXME: GCC and ICC form a J...E mangling for the pack here. Clang
    // doesn't do so when mangling an <unresolved-prefix>. It's not clear who's
    // right. See https://github.com/itanium-cxx-abi/cxx-abi/issues/113.
    // CHECK: @_ZN33partially_dependent_template_args5test11fIiEENS0_6enableIXsr4and_IT_S3_S3_EE5valueEE4typeES3_
    void g() { f(0); }
  }

  namespace test2 {
    struct X { int n; };
    template<unsigned> int f(X);

    template<typename T> void g1(decltype(f<0>(T()))) {}
    template<typename T> void g2(decltype(f<0>({}) + T())) {}
    template<typename T> void g3(decltype(f<0>(X{}) + T())) {}
    template<int N> void g4(decltype(f<0>(X{N})));

    // The first of these mangles the unconverted argument Li0E because the
    // callee is unresolved, the rest mangle the converted argument Lj0E
    // because the callee is resolved.
    void h() {
      // CHECK: @_ZN33partially_dependent_template_args5test22g1INS0_1XEEEvDTcl1fILi0EEcvT__EEE
      g1<X>({});
      // CHECK: @_ZN33partially_dependent_template_args5test22g2IiEEvDTplclL_ZNS0_1fILj0EEEiNS0_1XEEilEEcvT__EE
      g2<int>({});
      // CHECK: @_ZN33partially_dependent_template_args5test22g3IiEEvDTplclL_ZNS0_1fILj0EEEiNS0_1XEEtlS3_EEcvT__EE
      g3<int>({});
      // CHECK: @_ZN33partially_dependent_template_args5test22g4ILi0EEEvDTclL_ZNS0_1fILj0EEEiNS0_1XEEtlS3_T_EEE
      g4<0>({});
    }
  }
}

namespace fixed_size_parameter_pack {
  template<typename ...T> struct A {
    template<T ...> struct B {};
  };
  template<int ...Ns> void f(A<unsigned, char, long long>::B<0, Ns...>);
  void g() { f<1, 2>({}); }
}

namespace type_qualifier {
  template<typename T> using int_t = int;
  template<typename T> void f(decltype(int_t<T*>() + 1)) {}
  // FIXME: This mangling doesn't work: we need to mangle the
  // instantiation-dependent 'int_t' operand.
  // CHECK: @_ZN14type_qualifier1fIPiEEvDTplcvi_ELi1EE
  template void f<int*>(int);

  // Note that this template has different constraints but would mangle the
  // same:
  //template<typename T> void f(decltype(int_t<typename T::type>() + 1)) {}

  struct impl { using type = void; };
  template<typename T> using alias = impl;
  template<typename T> void g(decltype(alias<T*>::type(), 1)) {}
  // FIXME: Similarly we need to mangle the `T*` in here.
  // CHECK: @_ZN14type_qualifier1gIPiEEvDTcmcvv_ELi1EE
  template void g<int*>(int);
}
