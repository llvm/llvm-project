// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -fvisibility=hidden -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-HIDDEN
// For clang, "internal" is just an alias for "hidden". We could use it for some
// optimization purposes on 32-bit x86, but it's not worth it.
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -fvisibility=internal -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-HIDDEN

#define HIDDEN __attribute__((visibility("hidden")))
#define PROTECTED __attribute__((visibility("protected")))
#define DEFAULT __attribute__((visibility("default")))

namespace test30 {
  // When H is hidden, it should make X hidden, even if the template argument
  // is not.
  struct H {
  };
  template<H *T>
  struct X {
  };
  H DEFAULT a;
  X<&a> b;
  // CHECK: _ZN6test301bE = global
  // CHECK-HIDDEN: _ZN6test301bE = hidden global
}

namespace test25 {
  template<typename T>
  struct X {
    template<typename U>
    struct definition {
    };
  };

  class DEFAULT A { };

  X<int>::definition<A> a;
  // CHECK: @_ZN6test251aE = global
  // CHECK-HIDDEN: @_ZN6test251aE = hidden global
}

namespace test28 {
  class DEFAULT foo {
  };
  foo myvec;
  // CHECK: @_ZN6test285myvecE = global
  // CHECK-HIDDEN: @_ZN6test285myvecE = hidden global
}

namespace test29 {
#pragma GCC visibility push(hidden)
  struct RECT {
    int top;
  };
  DEFAULT extern RECT data_rect;
  RECT data_rect = { -1};
#pragma GCC visibility pop
  // CHECK: @_ZN6test299data_rectE = global
  // CHECK-HIDDEN: @_ZN6test299data_rectE = global
}

namespace test40 {
  template<typename T>
  struct foo {
    DEFAULT static int bar;
  };
  template<typename T>
  int foo<T>::bar;
  template struct foo<int>;
  // CHECK: _ZN6test403fooIiE3barE = weak_odr global
  // CHECK-HIDDEN: _ZN6test403fooIiE3barE = weak_odr global
}

namespace test41 {
  // Unlike gcc we propagate the information that foo not only is hidden, but
  // has been explicitly marked as so. This lets us produce a hidden undefined
  // reference to bar.
  struct HIDDEN foo {};
  extern foo bar;
  foo *zed() {
    return &bar;
  }
  // CHECK: @_ZN6test413barE = external hidden global
  // CHECK-HIDDEN: @_ZN6test413barE = external hidden global
}

namespace test48 {
  // Test that we use the visibility of struct foo when instantiating the
  // template. Note that is a case where we disagree with gcc, it produces
  // a default symbol.
  struct HIDDEN foo {
  };
  DEFAULT foo x;

  struct bar {
    template<foo *z>
    struct zed {
    };
  };

  bar::zed<&x> y;
  // CHECK: _ZN6test481yE = hidden global
  // CHECK-HIDDEN: _ZN6test481yE = hidden global
}

namespace test72 {
  template <class T>
  struct foo {
    HIDDEN static int var1;
    template <class U> HIDDEN static U var2;
  };
  template <class T> template <class U>
  U foo<T>::var2;

  extern template struct DEFAULT foo<int>;

  int use() {
    foo<int> o;
    foo<long> p;
    return o.var1 + o.var2<int> + p.var1 + p.var2<int>;
  }
  // CHECK:      @_ZN6test723fooIiE4var1E = external hidden global i32
  // CHECK-NEXT: @_ZN6test723fooIiE4var2IiEE = linkonce_odr global i32 0
  // CHECK-NEXT: @_ZN6test723fooIlE4var1E = external hidden global i32
  // CHECK-NEXT: @_ZN6test723fooIlE4var2IiEE = linkonce_odr global i32 0
}

namespace test73 {
  struct HIDDEN foo {};
  DEFAULT foo da, db, dc, dd;
  HIDDEN foo ha, hb, hc, hd;
  template<foo *z> DEFAULT int var;

  template int var<&da>;
  template int DEFAULT var<&db>;
  template int HIDDEN var<&dc>;
  template int var<&ha>;
  template int DEFAULT var<&hb>;
  template int HIDDEN var<&hc>;

  int use() { return var<&dd> + var<&hd>; }
  // CHECK:      @_ZN6test733varIXadL_ZNS_2daEEEEE = weak_odr hidden global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2dbEEEEE = weak_odr global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2dcEEEEE = weak_odr hidden global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2haEEEEE = weak_odr hidden global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2hbEEEEE = weak_odr global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2hcEEEEE = weak_odr hidden global i32 0
  // CHECK:      @_ZN6test733varIXadL_ZNS_2ddEEEEE = linkonce_odr hidden global i32 0
  // CHECK-NEXT: @_ZN6test733varIXadL_ZNS_2hdEEEEE = linkonce_odr hidden global i32 0

  // CHECK-HIDDEN:      @_ZN6test733varIXadL_ZNS_2daEEEEE = weak_odr hidden global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2dbEEEEE = weak_odr global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2dcEEEEE = weak_odr hidden global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2haEEEEE = weak_odr hidden global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2hbEEEEE = weak_odr global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2hcEEEEE = weak_odr hidden global i32 0
  // CHECK-HIDDEN:      @_ZN6test733varIXadL_ZNS_2ddEEEEE = linkonce_odr hidden global i32 0
  // CHECK-HIDDEN-NEXT: @_ZN6test733varIXadL_ZNS_2hdEEEEE = linkonce_odr hidden global i32 0
}

// CHECK: @_ZN5Test425VariableInHiddenNamespaceE = hidden global i32 10
// CHECK: @_ZN5Test71aE = hidden global
// CHECK: @_ZN5Test71bE = global
// CHECK: @test9_var = global
// CHECK-HIDDEN: @test9_var = global
// CHECK: @_ZN6Test121A6hiddenE = external hidden global
// CHECK: @_ZN6Test121A7visibleE = external global
// CHECK-HIDDEN: @_ZN6Test121A6hiddenE = external hidden global
// CHECK-HIDDEN: @_ZN6Test121A7visibleE = external global
// CHECK: @_ZN6Test131B1aE = hidden global
// CHECK: @_ZN6Test131C1aE = global
// CHECK-HIDDEN: @_ZN6Test131B1aE = hidden global
// CHECK-HIDDEN: @_ZN6Test131C1aE = global
// CHECK: @_ZN6Test143varE = external global
// CHECK-HIDDEN: @_ZN6Test143varE = external global
// CHECK: @_ZN6Test154TempINS_1AEE5Inner6bufferE = external global [0 x i8]
// CHECK-HIDDEN: @_ZN6Test154TempINS_1AEE5Inner6bufferE = external global [0 x i8]
// CHECK: @_ZTVN6test701BE = external hidden unnamed_addr constant { [5 x ptr] }, align 8
// CHECK: @_ZTTN6test701BE = external hidden unnamed_addr constant [2 x ptr], align 8

namespace test27 {
  template<typename T>
  class C {
    class DEFAULT D {
      void f();
    };
  };

  template<>
  class C<int>::D {
    virtual void g();
  };

  void C<int>::D::g() {
  }
  // CHECK: _ZTVN6test271CIiE1DE = unnamed_addr constant
  // CHECK-HIDDEN: _ZTVN6test271CIiE1DE = unnamed_addr constant
}

// CHECK: @_ZTVN5Test63fooE = linkonce_odr hidden unnamed_addr constant

// CHECK-HIDDEN: @_ZTVN6Test161AIcEE = external unnamed_addr constant
// CHECK-HIDDEN: @_ZTTN6Test161AIcEE = external unnamed_addr constant

// CHECK: @_ZZN6test681fC1EvE4test = linkonce_odr global
// CHECK-HIDDEN: @_ZZN6test681fC1EvE4test = linkonce_odr hidden global

// CHECK: @_ZGVZN6test681fC1EvE4test = linkonce_odr global
// CHECK-HIDDEN: @_ZGVZN6test681fC1EvE4test = linkonce_odr hidden global

// CHECK-HIDDEN: @_ZTVN6test701DE = linkonce_odr hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZTIN6test701DE] }, align 8
// CHECK-HIDDEN: @_ZTTN6test701DE = linkonce_odr hidden unnamed_addr constant [1 x ptr] [ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTVN6test701DE, i32 0, i32 0, i32 3)], align 8

// CHECK: @_ZZN6Test193fooIiEEvvE1a = linkonce_odr global
// CHECK-HIDDEN: @_ZZN6Test193fooIiEEvvE1a = linkonce_odr hidden global

// CHECK: @_ZGVZN6Test193fooIiEEvvE1a = linkonce_odr global i64
// CHECK-HIDDEN: @_ZGVZN6Test193fooIiEEvvE1a = linkonce_odr hidden global i64

namespace Test1 {
  // CHECK-LABEL: define hidden void @_ZN5Test11fEv
  void HIDDEN f() { }
  
}

namespace Test2 {
  struct HIDDEN A {
    void f();
  };

  // A::f is a member function of a hidden class.
  // CHECK-LABEL: define hidden void @_ZN5Test21A1fEv
  void A::f() { }
}
 
namespace Test3 {
  struct HIDDEN A {
    struct B {
      void f();
    };
  };

  // B is a nested class where its parent class is hidden.
  // CHECK-LABEL: define hidden void @_ZN5Test31A1B1fEv
  void A::B::f() { }  
}

namespace Test4 HIDDEN {
  int VariableInHiddenNamespace = 10;

  // Test4::g is in a hidden namespace.
  // CHECK-LABEL: define hidden void @_ZN5Test41gEv
  void g() { } 
  
  struct DEFAULT A {
    void f();
  };
  
  // A has default visibility.
  // CHECK-LABEL: define void @_ZN5Test41A1fEv
  void A::f() { } 
}

namespace Test5 {

  namespace NS HIDDEN {
    // f is in NS which is hidden.
    // CHECK-LABEL: define hidden void @_ZN5Test52NS1fEv()
    void f() { }
  }
  
  namespace NS {
    // g is in NS, but this NS decl is not hidden.
    // CHECK-LABEL: define void @_ZN5Test52NS1gEv
    void g() { }
  }
}

namespace Test6 {
  struct HIDDEN foo {
    foo() { }
    void bonk();
    virtual void bar() = 0;

    virtual void zonk() {}
  };

  struct barc : public foo {
    barc();
    virtual void bar();
  };

  barc::barc() {}
}

namespace Test7 {
  class HIDDEN A {};
  A a; // top of file

  template <A&> struct Aref {
    static void foo() {}
  };

  class B : public A {};
  B b; // top of file

  // CHECK-LABEL: define linkonce_odr hidden void @_ZN5Test74ArefIL_ZNS_1aEEE3fooEv()
  void test() {
    Aref<a>::foo();
  }
}

namespace Test8 {
  void foo();
  void bar() {}
  // CHECK-HIDDEN-LABEL: define hidden void @_ZN5Test83barEv()
  // CHECK-HIDDEN: declare void @_ZN5Test83fooEv()

  void test() {
    foo();
    bar();
  }
}

// PR8457
namespace Test9 {
  extern "C" {
    struct A { int field; };
    void DEFAULT test9_fun(struct A *a) { }
    struct A DEFAULT test9_var; // above
  }
  // CHECK-LABEL: define void @test9_fun(
  // CHECK-HIDDEN-LABEL: define void @test9_fun(

  void test() {
    A a = test9_var;
    test9_fun(&a);
  }
}

// PR8478
namespace Test10 {
  struct A;

  class DEFAULT B {
    void foo(A*);
  };

  // CHECK-LABEL: define void @_ZN6Test101B3fooEPNS_1AE(
  // CHECK-HIDDEN-LABEL: define void @_ZN6Test101B3fooEPNS_1AE(
  void B::foo(A*) {}
}

// PR8492
namespace Test11 {
  struct A {
    void foo() {}
    void DEFAULT bar() {}
  };

  void test() {
    A a;
    a.foo();
    a.bar();
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN6Test111A3fooEv(
  // CHECK-LABEL: define linkonce_odr void @_ZN6Test111A3barEv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6Test111A3fooEv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr void @_ZN6Test111A3barEv(
}

// Tested at top of file.
namespace Test12 {
  struct A {
    // This is hidden in all cases: the explicit attribute takes
    // priority over -fvisibility on the parent.
    static int hidden HIDDEN;

    // This is default in all cases because it's only a declaration.
    static int visible;
  };

  void test() {
    A::hidden = 0;
    A::visible = 0;
  }
}

// Tested at top of file.
namespace Test13 {
  struct HIDDEN A {};

  // Should be hidden in all cases.
  struct B {
    static A a;
  };
  A B::a;

  // Should be default in all cases.
  struct DEFAULT C {
    static A a;
  };
  A C::a;
};

// Tested at top of file.
namespace Test14 {
  // Neither the visibility of the type nor -fvisibility=hidden should
  // apply to declarations.
  extern struct A *var;

  struct A *test() { return var; }
}

namespace Test15 {
  struct A {};
  template <class T> struct Temp {
    struct Inner {
      static char buffer[0];
    };
  };

  char *test() {
    return Temp<A>::Inner::buffer;
  }
}

namespace Test16 {
  struct Base1 { virtual void foo(); };
  struct Base2 : virtual Base1 { virtual void foo(); };
  template <class T> struct A : virtual Base1, Base2 {
    virtual void foo();
  };
  extern template struct A<char>;

  void test() {
    A<char> a;
    a.foo();
  }
}

namespace Test17 {
  struct HIDDEN A {
    static void foo();
    static void DEFAULT bar();
    static void HIDDEN baz();

    struct DEFAULT B {
      static void foo();
      static void DEFAULT bar();
      static void HIDDEN baz();
    };
  };

  void test() {
    A::foo();
    A::bar();
    A::baz();
    A::B::foo();
    A::B::bar();
    A::B::baz();
  }
  // CHECK: declare hidden void @_ZN6Test171A3fooEv()
  // CHECK: declare void @_ZN6Test171A3barEv()
  // CHECK: declare hidden void @_ZN6Test171A3bazEv()
  // CHECK: declare void @_ZN6Test171A1B3fooEv()
  // CHECK: declare void @_ZN6Test171A1B3barEv()
  // CHECK: declare hidden void @_ZN6Test171A1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A3bazEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A1B3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test171A1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test171A1B3bazEv()
}

namespace Test18 {
  template <class T> struct HIDDEN A {
    static void foo();
    static void DEFAULT bar();
    static void HIDDEN baz();

    struct DEFAULT B {
      static void foo();
      static void DEFAULT bar();
      static void HIDDEN baz();
    };
  };
  struct HIDDEN H;

  void test() {
    A<int>::foo();
    A<int>::bar();
    A<int>::baz();
    A<int>::B::foo();
    A<int>::B::bar();
    A<int>::B::baz();
    A<H>::foo();
    A<H>::bar();
    A<H>::baz();
    A<H>::B::foo();
    A<H>::B::bar();
    A<H>::B::baz();
  }
  // CHECK: declare hidden void @_ZN6Test181AIiE3fooEv()
  // CHECK: declare void @_ZN6Test181AIiE3barEv()
  // CHECK: declare hidden void @_ZN6Test181AIiE3bazEv()
  // CHECK: declare void @_ZN6Test181AIiE1B3fooEv()
  // CHECK: declare void @_ZN6Test181AIiE1B3barEv()
  // CHECK: declare hidden void @_ZN6Test181AIiE1B3bazEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3fooEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3barEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE3bazEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3fooEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3barEv()
  // CHECK: declare hidden void @_ZN6Test181AINS_1HEE1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE3bazEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE1B3fooEv()
  // CHECK-HIDDEN: declare void @_ZN6Test181AIiE1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AIiE1B3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3fooEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE3bazEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3fooEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3barEv()
  // CHECK-HIDDEN: declare hidden void @_ZN6Test181AINS_1HEE1B3bazEv()
}

namespace Test19 {
  struct A { A(); ~A(); };

  // Tested at top of file.
  template <class T> void foo() {
    static A a;
  }

  void test() {
    foo<int>();
  }
}

// Various things with class template specializations.
namespace Test20 {
  template <unsigned> struct HIDDEN A {};

  // An explicit specialization inherits the explicit visibility of
  // the template.
  template <> struct A<0> {
    static void test0();
    static void test1();
  };

  // CHECK-LABEL: define hidden void @_ZN6Test201AILj0EE5test0Ev()
  void A<0>::test0() {}

  // CHECK: declare hidden void @_ZN6Test201AILj0EE5test1Ev()
  void test1() {
    A<0>::test1();
  }

  // ...unless that's explicitly overridden.
  template <> struct DEFAULT A<1> {
    static void test2();
    static void test3();
  };

  // CHECK-LABEL: define void @_ZN6Test201AILj1EE5test2Ev()
  void A<1>::test2() {}

  // CHECK: declare void @_ZN6Test201AILj1EE5test3Ev()
  void test3() {
    A<1>::test3();
  }

  // But we should assume that an unknown specialization has the
  // explicit visibility settings of the template.
  template <class T> struct B {
    static void test4() {}
    static void test5();
  };

  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6Test201BINS_1AILj2EEEE5test4Ev()
  void test4() {
    B<A<2> >::test4();
  }

  // CHECK: declare hidden void @_ZN6Test201BINS_1AILj2EEEE5test5Ev()
  void test5() {
    B<A<2> >::test5();
  }
}

// PR9371
namespace test21 {
  enum En { en };
  template<En> struct A {
    DEFAULT void foo() {}
  };

  // CHECK-LABEL: define weak_odr void @_ZN6test211AILNS_2EnE0EE3fooEv(
  template void A<en>::foo();
}

// Visibility on explicit specializations should take precedence.
namespace test22 {
  class A1 {};
  class A2 {};

  template <class T> struct B {};
  template <> struct DEFAULT B<A1> {
    static void foo();
    static void bar() {}
  };
  template <> struct B<A2> {
    static void foo();
    static void bar() {}
  };

  void test() {
    B<A1>::foo();
    B<A1>::bar();
    B<A2>::foo();
    B<A2>::bar();
  }
  // CHECK: declare void @_ZN6test221BINS_2A1EE3fooEv()
  // CHECK-LABEL: define linkonce_odr void @_ZN6test221BINS_2A1EE3barEv()
  // CHECK: declare void @_ZN6test221BINS_2A2EE3fooEv()
  // CHECK-LABEL: define linkonce_odr void @_ZN6test221BINS_2A2EE3barEv()
  // CHECK-HIDDEN: declare void @_ZN6test221BINS_2A1EE3fooEv()
  // CHECK-HIDDEN-LABEL: define linkonce_odr void @_ZN6test221BINS_2A1EE3barEv()
  // CHECK-HIDDEN: declare void @_ZN6test221BINS_2A2EE3fooEv()
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test221BINS_2A2EE3barEv()
}

namespace PR10113 {
  namespace foo DEFAULT {
    template<typename T>
      class bar {
      void zed() {}
    };
  }
  template class foo::bar<char>;
  // CHECK-LABEL: define weak_odr void @_ZN7PR101133foo3barIcE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN7PR101133foo3barIcE3zedEv

  struct zed {
  };
  template class foo::bar<zed>;
  // CHECK-LABEL: define weak_odr void @_ZN7PR101133foo3barINS_3zedEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN7PR101133foo3barINS_3zedEE3zedEv
}

namespace PR11690 {
  template<class T> struct Class {
    void size() const {
    }
  };
  template class DEFAULT Class<char>;
  // CHECK-LABEL: define weak_odr void @_ZNK7PR116905ClassIcE4sizeEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZNK7PR116905ClassIcE4sizeEv

  template<class T> void Method() {}
  template  DEFAULT void Method<char>();
  // CHECK-LABEL: define weak_odr void @_ZN7PR116906MethodIcEEvv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN7PR116906MethodIcEEvv
}

namespace PR11690_2 {
  namespace foo DEFAULT {
    class bar;
    template<typename T1, typename T2 = bar>
    class zed {
      void bar() {
      }
    };
  }
  struct baz {
  };
  template class foo::zed<baz>;
  // CHECK-LABEL: define weak_odr void @_ZN9PR11690_23foo3zedINS_3bazENS0_3barEE3barEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN9PR11690_23foo3zedINS_3bazENS0_3barEE3barEv
}

namespace test23 {
  // Having a template argument that is explicitly visible should not make
  // the template instantiation visible.
  template <typename T>
  struct X {
    static void f() {
    }
  };

  class DEFAULT A;

  void g() {
    X<A> y;
    y.f();
  }
  // CHECK-LABEL: define linkonce_odr void @_ZN6test231XINS_1AEE1fEv
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test231XINS_1AEE1fEv
}

namespace PR12001 {
  template <typename P1>
  void Bind(const P1& p1) {
  }

  class DEFAULT Version { };

  void f() {
    Bind(Version());
  }
  // CHECK-LABEL: define linkonce_odr void @_ZN7PR120014BindINS_7VersionEEEvRKT_
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN7PR120014BindINS_7VersionEEEvRKT_
}

namespace test24 {
  class DEFAULT A { };

  struct S {
    template <typename T>
    void mem() {}
  };

  void test() {
    S s;
    s.mem<A>();
  }
  // CHECK-LABEL: define linkonce_odr void @_ZN6test241S3memINS_1AEEEvv
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test241S3memINS_1AEEEvv
}

namespace test26 {
  template<typename T>
  class C {
    DEFAULT  void f();
  };

  template<>
  void C<int>::f() { }

  // CHECK-LABEL: define void @_ZN6test261CIiE1fEv
  // CHECK-HIDDEN-LABEL: define void @_ZN6test261CIiE1fEv
}

namespace test31 {
  struct A {
    struct HIDDEN B {
      static void DEFAULT baz();
    };
  };
  void f() {
    A::B::baz();
  }
  // CHECK: declare void @_ZN6test311A1B3bazEv()
  // CHECK-HIDDEN: declare void @_ZN6test311A1B3bazEv()
}

namespace test32 {
  struct HIDDEN A {
    struct DEFAULT B {
      void DEFAULT baz();
    };
  };
  void A::B::baz() {
  }
  // CHECK-LABEL: define void @_ZN6test321A1B3bazEv
  // CHECK-HIDDEN-LABEL: define void @_ZN6test321A1B3bazEv
}

namespace test33 {
  template<typename T>
  class foo {
    void bar() {}
  };
  struct HIDDEN zed {
  };
  template class DEFAULT foo<zed>;
  // CHECK-LABEL: define weak_odr void @_ZN6test333fooINS_3zedEE3barEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test333fooINS_3zedEE3barEv
}

namespace test34 {
  struct foo {
  };
  template<class T>
  void bar() {}
  template DEFAULT void bar<foo>();
  // CHECK-LABEL: define weak_odr void @_ZN6test343barINS_3fooEEEvv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test343barINS_3fooEEEvv
}

namespace test35 {
  // This is a really ugly testcase. GCC propagates the DEFAULT in zed's
  // definition. It's not really clear what we can do here, because we
  // produce the symbols before even seeing the DEFAULT definition of zed.
  // FIXME: Maybe the best thing to do here is error?  It's certainly hard
  // to argue that this ought to be valid.
  template<typename T>
  struct DEFAULT foo {
    void bar() {}
  };
  class zed;
  template class foo<zed>;
  class DEFAULT zed {
  };
  // CHECK-LABEL: define weak_odr void @_ZN6test353fooINS_3zedEE3barEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test353fooINS_3zedEE3barEv
}

namespace test36 {
  template<typename T1, typename T2>
  class foo {
    void bar() {}
  };
  class DEFAULT S1 {};
  struct HIDDEN S2 {};
  template class foo<S1, S2>;
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test363fooINS_2S1ENS_2S2EE3barEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test363fooINS_2S1ENS_2S2EE3barEv
}

namespace test37 {
  struct HIDDEN foo {
  };
  template<class T>
  DEFAULT void bar() {}
  template DEFAULT void bar<foo>();
  // CHECK-LABEL: define weak_odr void @_ZN6test373barINS_3fooEEEvv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test373barINS_3fooEEEvv
}

namespace test38 {
  template<typename T>
  class DEFAULT foo {
    void bar() {}
  };
  struct HIDDEN zed {
  };
  template class foo<zed>;
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test383fooINS_3zedEE3barEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test383fooINS_3zedEE3barEv
}

namespace test39 {
  class DEFAULT default_t;
  class HIDDEN hidden_t;
  template <class T> class A {
    template <class U> class B {
      HIDDEN void hidden() {}
      void noattr() {}
      template <class V> void temp() {}
    };
  };
  template class DEFAULT A<hidden_t>;
  template class DEFAULT A<hidden_t>::B<hidden_t>;
  template void A<hidden_t>::B<hidden_t>::temp<default_t>();
  template void A<hidden_t>::B<hidden_t>::temp<hidden_t>();

  // CHECK-LABEL: define weak_odr hidden void @_ZN6test391AINS_8hidden_tEE1BIS1_E6hiddenEv
  // CHECK-LABEL: define weak_odr void @_ZN6test391AINS_8hidden_tEE1BIS1_E6noattrEv
  // CHECK-LABEL: define weak_odr void @_ZN6test391AINS_8hidden_tEE1BIS1_E4tempINS_9default_tEEEvv

  // GCC produces a default for this one. Why?
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test391AINS_8hidden_tEE1BIS1_E4tempIS1_EEvv

  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test391AINS_8hidden_tEE1BIS1_E6hiddenEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test391AINS_8hidden_tEE1BIS1_E6noattrEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test391AINS_8hidden_tEE1BIS1_E4tempINS_9default_tEEEvv

  // GCC produces a default for this one. Why?
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test391AINS_8hidden_tEE1BIS1_E4tempIS1_EEvv
}

namespace test42 {
  struct HIDDEN foo {
  };
  template <class P>
  struct bar {
  };
  template <>
  struct HIDDEN bar<foo> {
    DEFAULT static void zed();
  };
  void bar<foo>::zed() {
  }
  // CHECK-LABEL: define void @_ZN6test423barINS_3fooEE3zedEv
  // CHECK-HIDDEN-LABEL: define void @_ZN6test423barINS_3fooEE3zedEv
}

namespace test43 {
  struct HIDDEN foo {
  };
  template <class P>
  void bar() {
  }
  template <>
  DEFAULT void bar<foo>() {
  }
  // CHECK-LABEL: define void @_ZN6test433barINS_3fooEEEvv
  // CHECK-HIDDEN-LABEL: define void @_ZN6test433barINS_3fooEEEvv
}

namespace test44 {
  template <typename T>
  struct foo {
    foo() {}
  };
  namespace {
    struct bar;
  }
  template struct DEFAULT foo<bar>;
  foo<bar> x;
  // CHECK-LABEL: define internal void @_ZN6test443fooINS_12_GLOBAL__N_13barEEC1Ev
  // CHECK-HIDDEN-LABEL: define internal void @_ZN6test443fooINS_12_GLOBAL__N_13barEEC1Ev
}

namespace test45 {
  template <typename T>
  struct foo {
    template <typename T2>
    struct bar {
      bar() {};
    };
  };
  namespace {
    struct zed;
  }
  template struct DEFAULT foo<int>::bar<zed>;
  foo<int>::bar<zed> x;
  // CHECK-LABEL: define internal void @_ZN6test453fooIiE3barINS_12_GLOBAL__N_13zedEEC1Ev
  // CHECK-HIDDEN-LABEL: define internal void @_ZN6test453fooIiE3barINS_12_GLOBAL__N_13zedEEC1Ev
}

namespace test46 {
  template <typename T>
  void foo() {
  }
  namespace {
    struct bar;
  }
  template DEFAULT void foo<bar>();
  void zed() {
    foo<bar>();
  }
  // CHECK-LABEL: define internal void @_ZN6test463fooINS_12_GLOBAL__N_13barEEEvv
  // CHECK-HIDDEN-LABEL: define internal void @_ZN6test463fooINS_12_GLOBAL__N_13barEEEvv
}

namespace test47 {
  struct foo {
    template <typename T>
    static void bar() {
    }
  };
  namespace {
    struct zed;
  }
  template DEFAULT void foo::bar<zed>();
  void baz() {
    foo::bar<zed>();
  }
  // CHECK-LABEL: define internal void @_ZN6test473foo3barINS_12_GLOBAL__N_13zedEEEvv
  // CHECK-HIDDEN-LABEL: define internal void @_ZN6test473foo3barINS_12_GLOBAL__N_13zedEEEvv
}

namespace test49 {
  // Test that we use the visibility of struct foo when instantiating the
  // template. Note that is a case where we disagree with gcc, it produces
  // a default symbol.

  struct HIDDEN foo {
  };

  DEFAULT foo x;

  struct bar {
    template<foo *z>
    void zed() {
    }
  };

  template void bar::zed<&x>();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test493bar3zedIXadL_ZNS_1xEEEEEvv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test493bar3zedIXadL_ZNS_1xEEEEEvv
}

namespace test50 {
  // Test that we use the visibility of struct foo when instantiating the
  // template. Note that is a case where we disagree with gcc, it produces
  // a default symbol.

  struct HIDDEN foo {
  };
  DEFAULT foo x;
  template<foo *z>
  struct DEFAULT bar {
    void zed() {
    }
  };
  template void bar<&x>::zed();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test503barIXadL_ZNS_1xEEEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test503barIXadL_ZNS_1xEEEE3zedEv
}

namespace test51 {
  // Test that we use the visibility of struct foo when instantiating the
  // template. Note that is a case where we disagree with gcc, it produces
  // a default symbol.

  struct HIDDEN foo {};
  DEFAULT foo da, db, dc, dd, de, df;
  HIDDEN foo ha, hb, hc, hd, he, hf;
  template<foo *z>
  void DEFAULT zed() {
  }
  template void zed<&da>();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2daEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2daEEEEEvv(

  template void DEFAULT zed<&db>();
  // CHECK-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2dbEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2dbEEEEEvv(

  template void HIDDEN zed<&dc>();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2dcEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2dcEEEEEvv(

  template void zed<&ha>();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2haEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2haEEEEEvv(

  template void DEFAULT zed<&hb>();
  // CHECK-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2hbEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2hbEEEEEvv(

  template void HIDDEN zed<&hc>();
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2hcEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2hcEEEEEvv(

#pragma GCC visibility push(hidden)
  template void zed<&dd>();
  template void zed<&hd>();
  template void DEFAULT zed<&he>();
#pragma GCC visibility pop
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2ddEEEEEvv(
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2hdEEEEEvv(
  // CHECK-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2heEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2ddEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr hidden void @_ZN6test513zedIXadL_ZNS_2hdEEEEEvv(
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test513zedIXadL_ZNS_2heEEEEEvv(

  void use() {
    zed<&df>();
    zed<&hf>();
  }
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test513zedIXadL_ZNS_2dfEEEEEvv(
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test513zedIXadL_ZNS_2hfEEEEEvv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test513zedIXadL_ZNS_2dfEEEEEvv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test513zedIXadL_ZNS_2hfEEEEEvv(
}

namespace test52 {
  // Test that we use the linkage of struct foo when instantiating the
  // template. Note that is a case where we disagree with gcc, it produces
  // an external symbol.

  namespace {
    struct foo {
    };
  }
  template<foo *x>
  void zed() {
  }
  void f() {
    zed<nullptr>();
  }
  // CHECK-LABEL: define internal void @_ZN6test523zedILPNS_12_GLOBAL__N_13fooE0EEEvv
  // CHECK-HIDDEN-LABEL: define internal void @_ZN6test523zedILPNS_12_GLOBAL__N_13fooE0EEEvv
}

namespace test53 {
  template<typename _Tp > struct vector   {
    static void       _M_fill_insert();
  };
#pragma GCC visibility push(hidden)
  // GCC doesn't seem to use the visibility of enums at all, we do.
  enum zed {v1};

  // GCC fails to mark this specialization hidden, we mark it.
  template<>
  struct vector<int> {
    static void       _M_fill_insert();
  };
  void foo() {
    vector<unsigned>::_M_fill_insert();
    vector<int>::_M_fill_insert();
    vector<zed>::_M_fill_insert();
  }
#pragma GCC visibility pop
  // CHECK: declare void @_ZN6test536vectorIjE14_M_fill_insertEv
  // CHECK-HIDDEN: declare void @_ZN6test536vectorIjE14_M_fill_insertEv
  // CHECK: declare hidden void @_ZN6test536vectorIiE14_M_fill_insertEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test536vectorIiE14_M_fill_insertEv
  // CHECK: declare hidden void @_ZN6test536vectorINS_3zedEE14_M_fill_insertEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test536vectorINS_3zedEE14_M_fill_insertEv
}

namespace test54 {
  template <class T>
  struct foo {
    static void bar();
  };
#pragma GCC visibility push(hidden)
  class zed {
    zed(const zed &);
  };
  void bah() {
    foo<zed>::bar();
  }
#pragma GCC visibility pop
  // CHECK: declare hidden void @_ZN6test543fooINS_3zedEE3barEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test543fooINS_3zedEE3barEv
}

namespace test55 {
  template <class T>
  struct HIDDEN foo {
    static void bar();
  };
  template <class T> struct foo;
  void foobar() {
    foo<int>::bar();
  }
  // CHECK: declare hidden void @_ZN6test553fooIiE3barEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test553fooIiE3barEv
}

namespace test56 {
  template <class T> struct foo;
  template <class T>
  struct HIDDEN foo {
    static void bar();
  };
  void foobar() {
    foo<int>::bar();
  }
  // CHECK: declare hidden void @_ZN6test563fooIiE3barEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test563fooIiE3barEv
}

namespace test57 {
#pragma GCC visibility push(hidden)
  template <class T>
  struct foo;
  void bar(foo<int>*);
  template <class T>
  struct foo {
    static void zed();
  };
  void bah() {
    foo<int>::zed();
  }
#pragma GCC visibility pop
  // CHECK: declare hidden void @_ZN6test573fooIiE3zedEv
  // CHECK-HIDDEN: declare hidden void @_ZN6test573fooIiE3zedEv
}

namespace test58 {
#pragma GCC visibility push(hidden)
  struct foo;
  template<typename T>
  struct DEFAULT bar {
    static void zed() {
    }
  };
  void bah() {
    bar<foo>::zed();
  }
#pragma GCC visibility pop
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test583barINS_3fooEE3zedEv
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test583barINS_3fooEE3zedEv
}

namespace test59 {
  DEFAULT int f();
  HIDDEN int g();
  typedef int (*foo)();
  template<foo x, foo y>
  void test() {}
  void use() {
    test<&g, &f>();
    // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test594testIXadL_ZNS_1gEvEEXadL_ZNS_1fEvEEEEvv
    // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test594testIXadL_ZNS_1gEvEEXadL_ZNS_1fEvEEEEvv

    test<&f, &g>();
    // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test594testIXadL_ZNS_1fEvEEXadL_ZNS_1gEvEEEEvv
    // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test594testIXadL_ZNS_1fEvEEXadL_ZNS_1gEvEEEEvv
  }
}

namespace test60 {
  template<int i>
  class HIDDEN a {};
  template<int i>
  class DEFAULT b {};
  template<template<int> class x, template<int> class y>
  void test() {}
  void use() {
    test<a, b>();
    // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test604testINS_1aENS_1bEEEvv
    // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test604testINS_1aENS_1bEEEvv

    test<b, a>();
    // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test604testINS_1bENS_1aEEEvv
    // CHECK-HIDDEN-LABEL: define linkonce_odr hidden void @_ZN6test604testINS_1bENS_1aEEEvv
  }
}

namespace test61 {
  template <typename T1>
  struct Class1
  {
    void f1() { f2(); }
    inline void f2();
  };
  template<>
  inline void Class1<int>::f2()
  {
  }
  void g(Class1<int> *x) {
    x->f1();
  }
}
namespace test61 {
  // Just test that we don't crash. Currently we apply this attribute. Current
  // gcc issues a warning about it being unused since "the type is already
  // defined". We should probably do the same.
  template class HIDDEN Class1<int>;
}

namespace test62 {
  template <typename T1>
  struct Class1
  {
    void f1() { f2(); }
    inline void f2() {}
  };
  template<>
  inline void Class1<int>::f2()
  {
  }
  void g(Class1<int> *x) {
    x->f2();
  }
}
namespace test62 {
  template class HIDDEN Class1<int>;
  // Just test that we don't crash. Currently we apply this attribute. Current
  // gcc issues a warning about it being unused since "the type is already
  // defined". We should probably do the same.
}

namespace test63 {
  enum HIDDEN E { E0 };
  struct A {
    template <E> static void foo() {}

    template <E> struct B {
      static void foo() {}
    };
  };

  void test() {
    A::foo<E0>();
    A::B<E0>::foo();
  }
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test631A3fooILNS_1EE0EEEvv()
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test631A1BILNS_1EE0EE3fooEv()
}

// Don't ignore the visibility of template arguments just because we
// explicitly instantiated something.
namespace test64 {
  struct HIDDEN A {};
  template <class P> struct B {
    static DEFAULT void foo() {}
  };

  template class B<A>;
  // CHECK-LABEL: define weak_odr hidden void @_ZN6test641BINS_1AEE3fooEv()
}

namespace test65 {
  class HIDDEN A {};
  template <class T> struct B {
    static void func();
    template <class U> static void funcT1();
    template <class U> static void funcT2();
    class Inner {};
    template <class U> class InnerT {};
  };
  template <template <class T> class Temp> struct C {
    static void foo() {}
  };

  // CHECK-LABEL: define void @_ZN6test651BINS_1AEE4funcEv()
  template <> DEFAULT void B<A>::func() {}

  // CHECK-LABEL: define void @_ZN6test651BINS_1AEE6funcT2IS1_EEvv()
  template <> template <> DEFAULT void B<A>::funcT2<A>() {}

  // CHECK-LABEL: define linkonce_odr void @_ZN6test651BINS_1AEE6funcT1IiEEvv()
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test651BINS_1AEE6funcT1IS1_EEvv()
  template <> template <class T> DEFAULT void B<A>::funcT1() {}

  // CHECK-LABEL: define linkonce_odr void @_ZN6test651BINS_1AEE5Inner3fooEv()
  template <> struct DEFAULT B<A>::Inner {
    static void foo() {}
  };

  // CHECK-LABEL: define linkonce_odr void @_ZN6test651BINS_1AEE6InnerTIiE3fooEv()
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN6test651BINS_1AEE6InnerTIS1_E3fooEv()
  template <> template <class U> struct DEFAULT B<A>::InnerT {
    static void foo() {}
  };

  void test() {
    B<A>::funcT1<int>();
    B<A>::funcT1<A>();
    B<A>::Inner::foo();
    B<A>::InnerT<int>::foo();
    B<A>::InnerT<A>::foo();
  }

  template class C<B<A>::InnerT>;
}

namespace test66 {
  template <typename T>
  struct DEFAULT barT {
    static void zed() {}
  };
  class foo;
  class DEFAULT foo;
  template struct barT<foo>;
  // CHECK-LABEL: define weak_odr void @_ZN6test664barTINS_3fooEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test664barTINS_3fooEE3zedEv

  template <int* I>
  struct DEFAULT barI {
    static void zed() {}
  };
  extern int I;
  extern int I DEFAULT;
  template struct barI<&I>;
  // CHECK-LABEL: define weak_odr void @_ZN6test664barIIXadL_ZNS_1IEEEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test664barIIXadL_ZNS_1IEEEE3zedEv

  typedef void (*fType)(void);
  template<fType F>
  struct DEFAULT barF {
    static void zed() {}
  };
  void F();
  void F() DEFAULT;
  template struct barF<F>;
  // CHECK-LABEL: define weak_odr void @_ZN6test664barFIXadL_ZNS_1FEvEEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test664barFIXadL_ZNS_1FEvEEE3zedEv
}

namespace test67 {
  template <typename T>
  struct DEFAULT bar {
    static void zed() {}
  };

  class foo;
  class compute {
    void f(foo *rootfoo);
  };
  class DEFAULT foo;

  template struct bar<foo>;
  // CHECK-LABEL: define weak_odr void @_ZN6test673barINS_3fooEE3zedEv
  // CHECK-HIDDEN-LABEL: define weak_odr void @_ZN6test673barINS_3fooEE3zedEv
}

namespace test68 {
  class A { public: ~A(); };
  class f {
  public:
    f() {
      static A test;
    }
  };
  void g() {
    f a;
  }
  // Check lines at top of file.
}

namespace test69 {
  // PR18174
  namespace foo {
    void f();
  }
  namespace foo {
    void f() {};
  }
  namespace foo __attribute__((visibility("hidden"))) {
  }
  // CHECK-LABEL: define void @_ZN6test693foo1fEv
  // CHECK-HIDDEN-LABEL: define hidden void @_ZN6test693foo1fEv
}

namespace test70 {
  // Make sure both the vtable and VTT declarations are marked "hidden"
  class HIDDEN A {
    virtual void a();
  };
  class HIDDEN B : virtual A {
    void a() override;
    ~B();
  };
  B::~B() {}

  // Make sure both the vtable and VTT declarations are marked "hidden"
  // when "-fvisibilty=hidden" is in use.
  class C {};
  class D : virtual C {};
  D d;
  // Check lines at top of file.
}

// https://github.com/llvm/llvm-project/issues/31462
namespace test71 {
  template <class T>
  struct foo {
    static HIDDEN T zed();
    template <class U> HIDDEN U bar();
  };
  template <class T>
  T foo<T>::zed() { return {}; }
  template <class T> template <class U>
  U foo<T>::bar() { return {}; }

  extern template struct DEFAULT foo<int>;

  int use() {
    foo<int> o;
    foo<long> p;
    return o.zed() + o.bar<int>() + p.zed() + p.bar<int>();
  }
  /// FIXME: foo<int>::bar is hidden in GCC w/ or w/o -fvisibility=hidden.
  // CHECK-LABEL: declare hidden noundef i32 @_ZN6test713fooIiE3zedEv(
  // CHECK-LABEL: define linkonce_odr noundef i32 @_ZN6test713fooIiE3barIiEET_v(
  // CHECK-LABEL: define linkonce_odr hidden noundef i64 @_ZN6test713fooIlE3zedEv(
  // CHECK-LABEL: define linkonce_odr noundef i32 @_ZN6test713fooIlE3barIiEET_v(
  // CHECK-HIDDEN-LABEL: declare hidden noundef i32 @_ZN6test713fooIiE3zedEv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr noundef i32 @_ZN6test713fooIiE3barIiEET_v(
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden noundef i64 @_ZN6test713fooIlE3zedEv(
  // CHECK-HIDDEN-LABEL: define linkonce_odr hidden noundef i32 @_ZN6test713fooIlE3barIiEET_v(
}
