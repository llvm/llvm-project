// RUN: %clang_cc1 -fsyntax-only -verify -Wno-non-c-typedef-for-linkage -std=gnu++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-non-c-typedef-for-linkage -Wno-c++11-extensions -Wno-local-type-template-args %s -std=gnu++98
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-non-c-typedef-for-linkage -Wno-c++11-extensions -Wno-local-type-template-args -fmodules %s

namespace test1 {
  int x; // expected-note {{previous definition is here}}
  static int y;
  void f() {} // expected-note {{previous definition is here}}

  extern "C" {
    extern int x; // expected-error {{declaration of 'x' has a different language linkage}}
    extern int y; // OK, has internal linkage, so no language linkage.
    void f(); // expected-error {{declaration of 'f' has a different language linkage}}
  }
}

// This is OK. Both test2_f don't have language linkage since they have
// internal linkage.
extern "C" {
  static void test2_f() {
  }
  static void test2_f(int x) {
  }
}

namespace test3 {
  extern "C" {
    namespace {
      extern int x2;
      void f2();
    }
  }
  namespace {
    int x2;
    void f2() {}
  }
}

namespace test4 {
  void dummy() {
    void Bar();
    class A {
      friend void Bar();
    };
  }
}

namespace test5 {
  static void g();
  void f()
  {
    void g();
  }
}

// pr14898
namespace test6 {
  template <class _Rp>
  class __attribute__ ((__visibility__("default"))) shared_future;
  template <class _Rp>
  class future {
    template <class> friend class shared_future;
    shared_future<_Rp> share();
  };
  template <class _Rp> future<_Rp>
  get_future();
  template <class _Rp>
  struct shared_future<_Rp&> {
    shared_future(future<_Rp&>&& __f);
  };
  void f() {
    typedef int T;
    get_future<int>();
    typedef int& U;
    shared_future<int&> f1 = get_future<int&>();
  }
}

// This is OK. The variables have internal linkage and therefore no language
// linkage.
extern "C" {
  static int test7_x;
}
extern "C++" {
  extern int test7_x;
}
extern "C++" {
  static int test7_y;
}
extern "C" {
  extern int test7_y;
}
extern "C" { typedef int test7_F(); static test7_F test7_f; }
extern "C++" { extern test7_F test7_f; }

// FIXME: This should be invalid. The function has no language linkage, but
// the function type has, so this is redeclaring the function with a different
// type.
extern "C++" {
  static void test8_f();
}
extern "C" {
  extern void test8_f();
}
extern "C" {
  static void test8_g();
}
extern "C++" {
  extern void test8_g();
}

extern "C" {
  void __attribute__((overloadable)) test9_f(int c); // expected-note {{previous declaration is here}}
}
extern "C++" {
  void __attribute__((overloadable)) test9_f(int c); // expected-error {{declaration of 'test9_f' has a different language linkage}}
}

extern "C" {
  void __attribute__((overloadable)) test10_f(int);
  void __attribute__((overloadable)) test10_f(double);
}

extern "C" {
  void test11_f() {
    void  __attribute__((overloadable)) test11_g(int);
    void  __attribute__((overloadable)) test11_g(double);
  }
}

namespace test12 {
  const int n = 0;
  extern const int n;
  void f() {
    extern const int n;
  }
}

namespace test13 {
  static void a(void);
  extern void a();
  static void a(void) {}
}

namespace test14 {
  // Anonymous namespace implies internal linkage, so 'static' has no effect.
  namespace {
    void a(void);
    static void a(void) {}
  }
}

namespace test15 {
  const int a = 5; // expected-note {{previous definition is here}}
  static const int a; // expected-error {{redefinition of 'a'}}
}

namespace test16 {
  extern "C" {
    class Foo {
      int x;
      friend int bar(Foo *y);
    };
    int bar(Foo *y) {
      return y->x;
    }
  }
}

namespace test17 {
  namespace {
    struct I {
    };
  }
  template <typename T1, typename T2> void foo() {}
  template <typename T, T x> void bar() {} // expected-note {{candidate function}}
  inline void *g() {
    struct L {
    };
    // foo<L, I>'s linkage should be the merge of UniqueExternalLinkage (or
    // InternalLinkage in c++11) and VisibleNoLinkage. The correct answer is
    // NoLinkage in both cases. This means that using foo<L, I> as a template
    // argument should fail.
    return reinterpret_cast<void*>(bar<typeof(foo<L, I>), foo<L, I> >); // expected-error {{reinterpret_cast cannot resolve overloaded function 'bar' to type 'void *}}
  }
  void h() {
    g();
  }
}

namespace test18 {
  template <typename T> struct foo {
    template <T *P> static void f() {}
    static void *g() { return (void *)f<&x>; }
    static T x;
  };
  template <typename T> T foo<T>::x;
  inline void *f() {
    struct S {
    };
    return foo<S>::g();
  }
  void *h() { return f(); }
}

extern "C" void pr16247_foo(int);
static void pr16247_foo(double);
void pr16247_foo(int) {}
void pr16247_foo(double) {}

namespace PR16247 {
  extern "C" void pr16247_bar(int);
  static void pr16247_bar(double);
  void pr16247_bar(int) {}
  void pr16247_bar(double) {}
}
namespace PR18964 {
  unsigned &*foo; //expected-error{{'foo' declared as a pointer to a reference of type}}
  extern struct {} *foo; // don't assert
}

namespace typedef_name_for_linkage {
  template<typename T> struct Use {};

  struct A { A(); A(const A&); ~A(); };

  typedef struct {
    A a;
  } B;

  struct C {
    typedef struct {
      A a;
    } D;
  };

  typedef struct {
    void f() { static int n; struct Inner {};}
  } E;

  // FIXME: Ideally this would be accepted in all modes. In C++98, we trigger a
  // linkage calculation to drive the "internal linkage type as template
  // argument" warning.
  typedef struct {
    void f() { struct Inner {}; Use<Inner> ui; }
  } F;
#if __cplusplus < 201103L
  // expected-error@-4 {{given name for linkage purposes by typedef declaration after its linkage was computed}}
  // expected-note@-4 {{due to this member}}
  // expected-note@-4 {{by this typedef}}
#endif
}
