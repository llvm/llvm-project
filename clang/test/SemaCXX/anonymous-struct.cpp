// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

struct S {
  S();
#if __cplusplus <= 199711L
  // expected-note@-2 {{because type 'S' has a user-provided default constructor}}
#endif
};

struct { // expected-error {{anonymous structs and classes must be class members}} expected-warning {{does not declare anything}}
};

struct E {
  struct {
    S x;
#if __cplusplus <= 199711L
    // expected-error@-2 {{anonymous struct member 'x' has a non-trivial default constructor}}
#endif
  };
  static struct { // expected-warning {{does not declare anything}}
  };
  class {
    int anon_priv_field; // expected-error {{anonymous struct cannot contain a private data member}}
  };
};

template <class T> void foo(T);
typedef struct { // expected-error {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration after its linkage was computed; add a tag name here to establish linkage prior to definition}}
// expected-note@-1 {{unnamed type used in template argument was declared here}}

  void test() { // expected-note {{type is not C-compatible due to this member declaration}}
    foo(this);
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
    // expected-note@-3 {{while substituting deduced template arguments}}
#endif
  }
} A; // expected-note {{type is given name 'A' for linkage purposes by this typedef declaration}}

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  int x = 0; // expected-note {{type is not C-compatible due to this default member initializer}} expected-warning 0-1{{extension}}
} B; // expected-note {{type is given name 'B' for linkage purposes by this typedef declaration}}

typedef struct // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
: B { // expected-note {{type is not C-compatible due to this base class}}
} C; // expected-note {{type is given name 'C' for linkage purposes by this typedef declaration}}

#if __cplusplus > 201703L && __cplusplus < 202002L
typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  static_assert([]{ return true; }()); // expected-note {{type is not C-compatible due to this lambda expression}}
} Lambda1; // expected-note {{type is given name 'Lambda1' for linkage purposes by this typedef declaration}}

template<int> struct X {};
typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  X<[]{ return 0; }()> x; // expected-note {{type is not C-compatible due to this lambda expression}}
  // FIXME: expected-error@-1 {{lambda expression cannot appear}}
} Lambda2; // expected-note {{type is given name 'Lambda2' for linkage purposes by this typedef declaration}}

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  enum E {
    a = []{ return 1; }() // expected-note {{type is not C-compatible due to this lambda expression}}
  };
} Lambda3; // expected-note {{type is given name 'Lambda3' for linkage purposes by this typedef declaration}}
#endif

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  template<int> void f() {} // expected-note {{type is not C-compatible due to this member declaration}}
} Template; // expected-note {{type is given name 'Template' for linkage purposes by this typedef declaration}}

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  struct U {
    void f(); // expected-note {{type is not C-compatible due to this member declaration}}
  };
} Nested; // expected-note {{type is given name 'Nested' for linkage purposes by this typedef declaration}}

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  friend void f() {} // expected-note {{type is not C-compatible due to this friend declaration}}
} Friend; // expected-note {{type is given name 'Friend' for linkage purposes by this typedef declaration}}

typedef struct { // expected-warning {{anonymous non-C-compatible type given name for linkage purposes by typedef declaration; add a tag name here}}
  template<typename T> friend void f() {} // expected-note {{type is not C-compatible due to this friend declaration}}
} FriendTemplate; // expected-note {{type is given name 'FriendTemplate' for linkage purposes by this typedef declaration}}

// Check that we don't diagnose the permitted cases:
typedef struct {
  // (non-members)
  _Static_assert(true, "");
  int : 0;
  /*empty-declaration*/;

  // non-static data members
  int a;
  // member enumerations
  enum E { x, y, z };
  // member classes
  struct S {};

  // recursively
  struct T { int a; };
} OK;

// There are still some known permitted cases that require an early linkage
// computation. Ensure we diagnose those too.
namespace ValidButUnsupported {
#if __cplusplus >= 201402L
  template<typename T> auto compute_linkage() {
    static int n;
    return &n;
  }

  typedef struct { // expected-error {{unsupported: anonymous type given name for linkage purposes by typedef declaration after its linkage was computed; add a tag name here to establish linkage}}
    struct X {};
    decltype(compute_linkage<X>()) a;
  } A; // expected-note {{by this typedef declaration}}
#endif

  // This fails in some language modes but not others.
  template<typename T> struct Y {
    static const int value = 10;
  };
  typedef struct { // expected-error 0-1{{unsupported}}
    enum X {};
    int arr[Y<X>::value];
  } B; // expected-note 0-1{{by this typedef}}

  template<typename T> void f() {}
  typedef struct { // expected-error {{unsupported}}
    enum X {};
    int arr[&f<X> ? 1 : 2];
#if __cplusplus < 201103L
    // expected-warning@-2 {{folded to constant}}
    // expected-warning@-3 {{variable length arrays in C++ are a Clang extension}}
#endif
  } C; // expected-note {{by this typedef}}
}

namespace ImplicitDecls {
struct Destructor {
  ~Destructor() {}
};
typedef struct {
} Empty;

typedef struct {
  Destructor x;
} A;

typedef struct {
  Empty E;
} B;

typedef struct {
  const Empty E;
} C;
} // namespace ImplicitDecls

struct {
  static int x; // expected-error {{static data member 'x' not allowed in anonymous struct}}
} static_member_1;

class {
  struct A {
    static int x; // expected-error {{static data member 'x' not allowed in anonymous class}}
  } x;
} static_member_2;

union {
  struct A {
    struct B {
      static int x; // expected-error {{static data member 'x' not allowed in anonymous union}}
    } x;
  } x;
} static_member_3;

// Ensure we don't compute the linkage of a member function just because it
// happens to have the same name as a builtin.
namespace BuiltinName {
  // Note that this is not an error: we didn't trigger linkage computation in this example.
  typedef struct { // expected-warning {{anonymous non-C-compatible type}}
    void memcpy(); // expected-note {{due to this member}}
  } A; // expected-note {{given name 'A' for linkage purposes by this typedef}}
}
namespace inline_defined_static_member {
typedef struct { // expected-warning {{anonymous non-C-compatible type}}
  static void f() { // expected-note {{due to this member}}
  }
} A; // expected-note {{given name 'A' for linkage purposes by this typedef}}
}

#if __cplusplus > 201103L
namespace GH58800 {
struct A {
  union {
    struct {
      float red = 0.0f;
    };
  };
};

A GetA() {
  A result{};
  return result;
}
}
#endif
