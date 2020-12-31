// RUN: %clang_cc1 %s -O1 -disable-llvm-passes -triple=x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O1 -disable-llvm-passes -triple=x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK2

// Instantiation order varies on different C++ dialects (IE, between C++98 and C++11).
// CHECK-DAG: @_ZN7PR100011xE ={{.*}} global
// CHECK-DAG: @_ZTVN5test018stdio_sync_filebufIA3_iEE = weak_odr unnamed_addr constant
// CHECK-DAG: @_ZN7PR100011SIiE3arrE = linkonce_odr global [3 x i32]
// CHECK-DAG: @_ZTVN5test018stdio_sync_filebufIA4_iEE = linkonce_odr unnamed_addr constant

// Negative checks go under prefix "CHECK2" to avoid interference with CHECK and CHECK-DAG.
// CHECK2-NOT: @_ZN7PR100014kBarE = external global i32
// CHECK2-NOT: @_ZTVN5test118stdio_sync_filebufIwEE ={{.*}} constant
// CHECK2-NOT: _ZTVN5test315basic_fstreamXXIcEE
// CHECK2-NOT: @_ZTVN5test018stdio_sync_filebufIA1_iEE
// CHECK2-NOT: @_ZTVN5test018stdio_sync_filebufIA2_iEE
// CHECK2-NOT: @_ZN7PR100011SIiE3arr2E = linkonce_odr global [3 x i32]A

// CHECK2-NOT: _ZTVN5test31SIiEE
// CHECK2-NOT: _ZTSN5test31SIiEE

// CHECK-LABEL: define linkonce_odr void @_ZN5test21CIiEC1Ev(%"class.test2::C"* {{[^,]*}} %this) unnamed_addr
// CHECK-LABEL: define linkonce_odr void @_ZN5test21CIiE6foobarIdEEvT_(
// CHECK-LABEL: define available_externally void @_ZN5test21CIiE6zedbarEd(

// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g1ENS_1SILi1EEE()
// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g1ENS_1SILi2EEE()
// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g1ENS_1SILi3EEE()
// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g2ENS_1SILi1EEE()
// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g2ENS_1SILi2EEE()
// CHECK-LABEL: define linkonce_odr void @_ZN7PR106662g2ENS_1SILi3EEE()
// CHECK: declare void @_ZN7PR106662h1ENS_1SILi1EEE()
// CHECK: declare void @_ZN7PR106662h1ENS_1SILi2EEE()
// CHECK: declare void @_ZN7PR106662h1ENS_1SILi3EEE()
// CHECK: declare void @_ZN7PR106662h2ENS_1SILi1EEE()
// CHECK: declare void @_ZN7PR106662h2ENS_1SILi2EEE()
// CHECK: declare void @_ZN7PR106662h2ENS_1SILi3EEE()

namespace test0 {
  struct  basic_streambuf   {
    virtual       ~basic_streambuf();
  };
  template<typename _CharT >
  struct stdio_sync_filebuf : public basic_streambuf {
    virtual void      xsgetn();
  };

  // This specialization is not a key function, so doesn't cause the vtable to
  // be instantiated unless we're instantiating a class definition anyway.
  template<> void stdio_sync_filebuf<int[1]>::xsgetn()  {
  }
  template<> void stdio_sync_filebuf<int[2]>::xsgetn()  {
  }
  template<> void stdio_sync_filebuf<int[3]>::xsgetn()  {
  }
  template<> void stdio_sync_filebuf<int[4]>::xsgetn()  {
  }
  extern template class stdio_sync_filebuf<int[2]>;

  // These two both cause vtables to be emitted.
  template class stdio_sync_filebuf<int[3]>;
  stdio_sync_filebuf<int[4]> implicit_instantiation;
}

namespace test1 {
  struct  basic_streambuf   {
    virtual       ~basic_streambuf();
  };
  template<typename _CharT >
  struct stdio_sync_filebuf : public basic_streambuf {
    virtual void      xsgetn();
  };

  // Just a declaration should not force the vtable to be emitted.
  template<> void stdio_sync_filebuf<wchar_t>::xsgetn();
}

namespace test2 {
  template<typename T1>
  class C {
  public:
    virtual ~C();
    void zedbar(double) {
    }
    template<typename T2>
    void foobar(T2 foo) {
    }
  };
  extern template class C<int>;
  void g() {
    // The extern template declaration should not prevent us from producing
    // the implicit constructor (test at the top).
    C<int> a;

    // or foobar(test at the top).
    a.foobar(0.0);

    // But it should prevent zebbar
    // (test at the top).
    a.zedbar(0.0);
  }
}

namespace test3 {
  template<typename T>
  class basic_fstreamXX  {
    virtual void foo(){}
    virtual void is_open() const  { }
  };

  extern template class basic_fstreamXX<char>;
  // This template instantiation should not cause us to produce a vtable.
  // (test at the top).
  template void basic_fstreamXX<char>::is_open() const;
}

namespace test3 {
  template <typename T>
  struct S  {
      virtual void m();
  };
  
  template<typename T>
  void S<T>::m() { }

  // Should not cause us to produce vtable because template instantiations
  // don't have key functions.
  template void S<int>::m();
}

namespace test4 {
  template <class T> struct A { static void foo(); };

  class B {
    template <class T> friend void A<T>::foo();
    B();
  };

  template <class T> void A<T>::foo() {
    B b;
  }

  unsigned test() {
    A<int>::foo();
  }
}

namespace PR8505 {
// Hits an assertion due to bogus instantiation of class B.
template <int i> class A {
  class B* g;
};
class B {
  void f () {}
};
// Should not instantiate class B since it is introduced in namespace scope.
// CHECK2-NOT: _ZN6PR85051AILi0EE1B1fEv
template class A<0>;
}

// Ensure that when instantiating initializers for static data members to
// complete their type in an unevaluated context, we *do* emit initializers with
// side-effects, but *don't* emit initializers and variables which are otherwise
// unused in the program.
namespace PR10001 {
  template <typename T> struct S {
    static const int arr[];
    static const int arr2[];
    static const int x, y;
    static int f();
  };

  extern int foo();
  extern int kBar;

  template <typename T> const int S<T>::arr[] = { 1, 2, foo() }; // possible side effects
  template <typename T> const int S<T>::arr2[] = { 1, 2, kBar }; // no side effects
  template <typename T> const int S<T>::x = sizeof(arr) / sizeof(arr[0]);
  template <typename T> const int S<T>::y = sizeof(arr2) / sizeof(arr2[0]);
  template <typename T> int S<T>::f() { return x + y; }

  int x = S<int>::f();
}

// Ensure that definitions are emitted for all friend functions defined within
// class templates. Order of declaration is extremely important here. Different
// instantiations of the class happen at different points during the deferred
// method body parsing and afterward. Those different points of instantiation
// change the exact form the class template appears to have.
namespace PR10666 {
  template <int N> struct S {
    void f1() { S<1> s; }
    friend void g1(S s) {}
    friend void h1(S s);
    void f2() { S<2> s; }
    friend void g2(S s) {}
    friend void h2(S s);
    void f3() { S<3> s; }
  };
  void test(S<1> s1, S<2> s2, S<3> s3) {
    g1(s1); g1(s2); g1(s3);
    g2(s1); g2(s2); g2(s3);
    h1(s1); h1(s2); h1(s3);
    h2(s1); h2(s2); h2(s3);
  }
}
