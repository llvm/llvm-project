// RUN: %clang_cc1 -verify -Wno-return-type -Wno-main -std=c++11 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
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
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIFivEXadL_ZNS_L1fEvEEEEvv(
    test<int(), &f>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIRFivEL_ZNS_L1fEvEEEvv(
    test<int(&)(), f>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIPKiXadL_ZNS_L1nEEEEEvv(
    test<const int*, &n>();
    // CHECK-LABEL: define internal {{.*}}void @_ZN6test124testIRKiL_ZNS_L1nEEEEvv(
    test<const int&, n>();
  }
}

// rdar://problem/12072531
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

