// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -Wno-unused-but-set-global -verify=no-global -std=c++17 %s
// no-global-no-diagnostics

static thread_local int tl_set_unused;  // expected-warning {{variable 'tl_set_unused' set but not used}}
static thread_local int tl_set_and_used;
thread_local int tl_no_static_set_unused;

// Warning should respect attributes.
[[maybe_unused]] static int with_maybe_unused;
__attribute__((unused)) static int with_unused_attr;

void f0() {
  tl_set_unused = 1;
  tl_set_and_used = 2;
  int x = tl_set_and_used;
  (void)x;

  tl_no_static_set_unused = 3;

  with_maybe_unused = 4;
  with_unused_attr = 5;
}

// Named namespace.
namespace test {
  static int set_unused;  // expected-warning {{variable 'set_unused' set but not used}}
  static int set_and_used;

  void f1() {
    set_unused = 1;
    set_and_used = 2;
    int x = set_and_used;
    (void)x;
  }
}

// Nested named namespace.
namespace outer {
namespace inner {
static int nested_unused; // expected-warning {{variable 'nested_unused' set but not used}}
void f2() {
  nested_unused = 5;
}
}
}

// Anonymous namespace (all vars have internal linkage).
namespace {
  int anon_ns_unused; // expected-warning {{variable 'anon_ns_unused' set but not used}}
  static int anon_ns_static_unused; // expected-warning {{variable 'anon_ns_static_unused' set but not used}}

  void f3() {
    anon_ns_unused = 1;
    anon_ns_static_unused = 2;
  }
}

// Function pointers at file scope.
static void (*unused_func_ptr)(); // expected-warning {{variable 'unused_func_ptr' set but not used}}
void SetUnusedCallback(void (*f)()) {
  unused_func_ptr = f;
}

static void (*used_func_ptr)();
void SetUsedCallback(void (*f)()) {
  used_func_ptr = f;
}
void CallUsedCallback() {
  if (used_func_ptr)
    used_func_ptr();
}

// Static data members (external linkage, should not warn).
class MyClass {
public:
  static int unused_static_member;
  static int used_static_member;
};

int MyClass::unused_static_member = 0;
int MyClass::used_static_member = 0;

void f4() {
  MyClass::unused_static_member = 10;

  MyClass::used_static_member = 20;
  int x = MyClass::used_static_member;
  (void)x;
}

// Static data members in a named namespace (external linkage, should not warn).
namespace named {
  struct NamedClass {
    static int w;
  };
  int NamedClass::w = 0;
}

void f5() {
  named::NamedClass::w = 4;
}

// Static data members in anonymous namespace (internal linkage, should warn).
namespace {
  class AnonClass {
  public:
    static int unused_member; // expected-warning {{variable 'unused_member' set but not used}}
    static int used_member;
  };

  int AnonClass::unused_member = 0;
  int AnonClass::used_member = 0;
}

void f6() {
  AnonClass::unused_member = 3;
  AnonClass::used_member = 4;
  int y = AnonClass::used_member;
  (void)y;
}

// Static data members in nested anonymous namespace (internal linkage, should warn).
namespace outer2 {
  namespace {
    struct NestedAnonClass {
      static int v; // expected-warning {{variable 'v' set but not used}}
    };
    int NestedAnonClass::v = 0;
  }
}

void f7() {
  outer2::NestedAnonClass::v = 5;
}

// Static data members set inside methods, read outside.
namespace {
  struct SetInMethod {
    static int x;
    static int y; // expected-warning {{variable 'y' set but not used}}
    void setX() { x = 1; }
    void setY() { y = 1; }
  };
  int SetInMethod::x;
  int SetInMethod::y;
}

void f8() {
  SetInMethod s;
  s.setX();
  s.setY();
  int v = SetInMethod::x;
  (void)v;  // only x is read
}

// External linkage static data members set inside methods.
struct ExtSetInMethod {
  static int x;
  void set() { x = 1; }
};
int ExtSetInMethod::x;

void f9() {
  ExtSetInMethod e;
  e.set();
}

// Regression tests for static data members of class templates in an
// anonymous namespace. namespace. Since every instantiation of a member are
// tracked as one, a read anywhere in the source counts as a use.
namespace {
  // Set in the template and read through an instantiation.
  template <int N> struct TmplCounted {
    static int live;
    TmplCounted() { ++live; }
  };
  template <int N> int TmplCounted<N>::live = 0;

  // Read only in an uninstantiated member function.
  template <int N> struct TmplUninstReader {
    static int x;
    void set() { x = 1; }
    int get() { return x; }
  };
  template <int N> int TmplUninstReader<N>::x = 0;

  // Set but never read in any of its instantiations, but diagnosed only once.
  template <int N> struct TmplSetOnly {
    static int x; // expected-warning {{variable 'x' set but not used}}
    void set() { x = 1; }
  };
  template <int N> int TmplSetOnly<N>::x = 0;

  // Explicit specializations get tracked separately from the template.
  template <int N> struct TmplExplicit {
    static int x;
    void set() { x = 1; }
  };
  template <int N> int TmplExplicit<N>::x = 0;
  template <> int TmplExplicit<1>::x = 1;
  template <> int TmplExplicit<2>::x = 2; // expected-warning {{variable 'x' set but not used}}
}

int f10() {
  TmplCounted<0> c;
  (void)c;
  TmplUninstReader<0>{}.set();
  TmplSetOnly<0>{}.set();
  TmplSetOnly<1>{}.set();
  TmplSetOnly<2>{}.set();
  TmplExplicit<0>{}.set();
  TmplExplicit<1>{}.set();
  TmplExplicit<2>{}.set();
  return TmplCounted<0>::live + TmplExplicit<0>::x + TmplExplicit<1>::x;
}

// FIXME: These are truly unused but we do not warn.
namespace {
  template <class T> struct TmplDepType {
    static T x;
    void set() { x = T(); }
  };
  template <class T> T TmplDepType<T>::x;

  template <int N> struct TmplSetViaInst {
    static int x;
  };
  template <int N> int TmplSetViaInst<N>::x = 0;
}

void f11() {
  TmplDepType<int>{}.set();
  TmplSetViaInst<0>::x = 1;
}
