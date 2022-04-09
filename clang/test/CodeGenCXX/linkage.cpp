// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -O1 -disable-llvm-passes %s -o - | FileCheck %s

namespace test1 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test11fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test2 {
  // CHECK-DAG: define internal void @_ZN5test21fIZNS_L1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  static inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test3 {
  // CHECK-DAG: define internal void @_ZN5test31fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test4 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test41fIZNS_1gILi1EEEPvvE1SEEvT_(
  template <typename T> void f(T) {}
  template <int N> inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  extern template void *g<1>();
  template void *g<1>();
}

namespace test5 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test51fIZNS_1gILi1EEEPvvE1SEEvT_(
  template <typename T> void f(T) {}
  template <int N> inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S>);
  }
  extern template void *g<1>();
  void *h() { return g<1>(); }
}

namespace test6 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test61fIZZNS_1gEvEN1S1hEvE1TEEvv(
  template <typename T> void f() {}

  inline void *g() {
    struct S {
      void *h() {
        struct T {
        };
        return (void *)f<T>;
      }
    } s;
    return s.h();
  }

  void *h() { return g(); }
}

namespace test7 {
  // CHECK-DAG: define internal void @_ZN5test71fIZZNS_1gEvEN1S1hEvE1TEEvv(
  template <typename T> void f() {}

  void *g() {
    struct S {
      void *h() {
        struct T {
        };
        return (void *)f<T>;
      }
    } s;
    return s.h();
  }

  void *h() { return g(); }
}

namespace test8 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test81fIZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    enum S {
    };
    return reinterpret_cast<void *>(f<S>);
  }
  void *h() { return g(); }
}

namespace test9 {
  // CHECK-DAG: define linkonce_odr void @_ZN5test91fIPZNS_1gEvE1SEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    return reinterpret_cast<void *>(f<S*>);
  }
  void *h() { return g(); }
}

namespace test10 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test101fIPFZNS_1gEvE1SvEEEvT_(
  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    } s;
    typedef S(*ftype)();
    return reinterpret_cast<void *>(f<ftype>);
  }
  void *h() { return g(); }
}

namespace test11 {
  // CHECK-DAG: define internal void @_ZN6test111fIPFZNS_1gEvE1SPNS_12_GLOBAL__N_11IEEEEvT_(
  namespace {
    struct I {
    };
  }

  template <typename T> void f(T) {}
  inline void *g() {
    struct S {
    };
    typedef S(*ftype)(I * x);
    return reinterpret_cast<void *>(f<ftype>);
  }
  void *h() { return g(); }
}

namespace test12 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test123fooIZNS_3barIZNS_3zedEvE2S2EEPvvE2S1EEvv
  template <typename T> void foo() {}
  template <typename T> inline void *bar() {
    enum S1 {
    };
    return reinterpret_cast<void *>(foo<S1>);
  }
  inline void *zed() {
    enum S2 {
    };
    return reinterpret_cast<void *>(bar<S2>);
  }
  void *h() { return zed(); }
}

namespace test13 {
  // CHECK-DAG: define linkonce_odr void @_ZZN6test133fooEvEN1S3barEv(
  inline void *foo() {
    struct S {
      static void bar() {}
    };
    return (void *)S::bar;
  }
  void *zed() { return foo(); }
}

namespace test14 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test143fooIZNS_1fEvE1SE3barILPS1_0EEEvv(
  template <typename T> struct foo {
    template <T *P> static void bar() {}
    static void *g() { return (void *)bar<nullptr>; }
  };
  inline void *f() {
    struct S {
    };
    return foo<S>::g();
  }
  void h() { f(); }
}

namespace test15 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test153zedIZNS_3fooIiEEPvvE3barEEvv(
  template <class T> void zed() {}
  template <class T> void *foo() {
    class bar {
    };
    return reinterpret_cast<void *>(zed<bar>);
  }
  void test() { foo<int>(); }
}

namespace test16 {
  // CHECK-DAG: define linkonce_odr void @_ZN6test163zedIZNS_3fooIiE3barEvE1SEEvv(
  template <class T> void zed() {}
  template <class T> struct foo {
    static void *bar();
  };
  template <class T> void *foo<T>::bar() {
    class S {
    };
    return reinterpret_cast<void *>(zed<S>);
  }
  void *test() { return foo<int>::bar(); }
}

namespace test17 {
  // CHECK-DAG: @_ZZN6test173fooILi42EEEPivE3bar = linkonce_odr
  // CHECK-DAG: define weak_odr noundef i32* @_ZN6test173fooILi42EEEPiv(
  template<int I>
  int *foo() {
    static int bar;
    return &bar;
  }
  template int *foo<42>();
}

// PR18408
namespace test18 {
  template<template<typename> class> struct A {};
  struct B { template<typename> struct C; };
  void f(A<B::C>) {}
  // CHECK-DAG: define void @_ZN6test181fENS_1AINS_1B1CEEE(
}
