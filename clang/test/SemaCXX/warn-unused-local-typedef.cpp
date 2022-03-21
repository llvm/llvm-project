// RUN: %clang_cc1 -fsyntax-only -Wunused-local-typedef -verify -std=c++1y %s

struct S {
  typedef int Foo;  // no diag
};

namespace N {
  typedef int Foo;  // no diag
  typedef int Foo2;  // no diag
}

template <class T> class Vec {};

typedef int global_foo;  // no diag

void f() {
  typedef int foo0;  // expected-warning {{unused typedef 'foo0'}}
  using foo0alias = int ;  // expected-warning {{unused type alias 'foo0alias'}}

  typedef int foo1 __attribute__((unused));  // no diag

  typedef int foo2;
  {
    typedef int foo2;  // expected-warning {{unused typedef 'foo2'}}
  }
  typedef foo2 foo3; // expected-warning {{unused typedef 'foo3'}}

  typedef int foo2_2;  // expected-warning {{unused typedef 'foo2_2'}}
  {
    typedef int foo2_2;
    typedef foo2_2 foo3_2; // expected-warning {{unused typedef 'foo3_2'}}
  }

  typedef int foo4;
  foo4 the_thing;

  typedef int* foo5;
  typedef foo5* foo6;  // no diag
  foo6 *myptr;

  struct S2 {
    typedef int Foo; // no diag
    typedef int Foo2; // expected-warning {{unused typedef 'Foo2'}}

    struct Deeper {
      typedef int DeepFoo;  // expected-warning {{unused typedef 'DeepFoo'}}
    };
  };

  S2::Foo s2foo;

  typedef struct {} foostruct; // expected-warning {{unused typedef 'foostruct'}}

  typedef struct {} foostruct2; // no diag
  foostruct2 fs2;

  typedef int vecint;  // no diag
  Vec<vecint> v;

  N::Foo nfoo;

  typedef int ConstExprInt;
  static constexpr int a = (ConstExprInt)4;
}

int printf(char const *, ...);

void test() {
  typedef signed long int superint; // no diag
  printf("%ld", (superint)42);

  typedef signed long int superint2; // no diag
  printf("%ld", static_cast<superint2>(42));

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
  typedef int trungl_bot_was_here; // no diag
#pragma clang diagnostic pop

  typedef int foo; // expected-warning {{unused typedef 'foo'}}
}

template <class T>
void template_fun(T t) {
  typedef int foo; // expected-warning {{unused typedef 'foo'}}
  typedef int bar; // no-diag
  bar asdf;

  struct S2 {
    typedef int Foo; // no diag

    typedef int Foo2; // expected-warning {{unused typedef 'Foo2'}}

    typedef int Foo3; // no diag
  };

  typename S2::Foo s2foo;
  typename T::Foo s3foo;

  typedef typename S2::Foo3 TTSF;  // expected-warning {{unused typedef 'TTSF'}}
}
void template_fun_user() {
  struct Local {
    typedef int Foo; // no-diag
    typedef int Bar; // expected-warning {{unused typedef 'Bar'}}
  } p;
  template_fun(p);
}

void typedef_in_nested_name() {
  typedef struct { // expected-warning {{add a tag name}}
    typedef int Foo; // expected-note {{}}
  } A; // expected-note {{}}
  A::Foo adsf;

  using A2 = struct { // expected-warning {{add a tag name}} expected-note {{this alias declaration}}
    typedef int Foo; // expected-note {{}}
  };
  A2::Foo adsf2;
}

auto sneaky() {
  struct S {
    // Local typedefs can be used after the scope they were in has closed:
    typedef int t;

    // Even if they aren't, this could be an inline function that could be used
    // in another TU, so this shouldn't warn either:
    typedef int s;

  private:
    typedef int p; // expected-warning{{unused typedef 'p'}}
  };
  return S();
}
auto x = sneaky();
decltype(x)::t y;

static auto static_sneaky() {
  struct S {
    typedef int t;
    // This function has internal linkage, so we can warn:
    typedef int s; // expected-warning {{unused typedef 's'}}
  };
  return S();
}
auto sx = static_sneaky();
decltype(sx)::t sy;

auto sneaky_with_friends() {
  struct S {
  private:
    friend class G;
    // Can't warn if we have friends:
    typedef int p;
  };
  return S();
}

namespace {
auto nstatic_sneaky() {
  struct S {
    typedef int t;
    // This function has internal linkage, so we can warn:
    typedef int s; // expected-warning {{unused typedef 's'}}
  };
  return S();
}
auto nsx = nstatic_sneaky();
decltype(nsx)::t nsy;
}

// Like sneaky(), but returning pointer to local type
template<typename T>
struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };
auto pointer_sneaky() {
  struct S {
    typedef int t;
    typedef int s;
  };
  return (S*)nullptr;
}
remove_reference<decltype(*pointer_sneaky())>::type::t py;

// Like sneaky(), but returning templated struct referencing local type.
template <class T> struct container { int a; T t; };
auto template_sneaky() {
  struct S {
    typedef int t;
    typedef int s;
  };
  return container<S>();
}
auto tx = template_sneaky();
decltype(tx.t)::t ty;

// Like sneaky(), but doing its sneakiness by returning a member function
// pointer.
auto sneaky_memfun() {
  struct S {
    typedef int type;
    int n;
  };
  return &S::n;
}

template <class T> void sneaky_memfun_g(int T::*p) {
  typename T::type X;
}

void sneaky_memfun_h() {
  sneaky_memfun_g(sneaky_memfun());
}

void typedefs_in_constructors() {
  struct A {};
  struct B : public A {
    // Neither of these two should warn:
    typedef A INHERITED;
    B() : INHERITED() {}

    typedef B SELF;
    B(int) : SELF() {}
  };
}

void *operator new(__SIZE_TYPE__, void *p) throw() { return p; }
void placement_new_and_delete() {
  struct MyStruct { };
  char memory[sizeof(MyStruct)];
  void *p = memory;

  typedef MyStruct A_t1;
  MyStruct *a = new (p) A_t1();

  typedef MyStruct A_t2;
  a->~A_t2();
}

namespace TypedefInLocalClassOfAMemberOfTemplateClass {
template<typename> struct A {
  void foo() {
    struct Inner {
      typedef int Int; // no-diag
      typedef char Char; // expected-warning {{unused typedef 'Char'}}
      Int m;
    } b;
  }
};

void foo() {
  A<int> x;
  x.foo();
}
} // TypedefInLocalClassOfTemplateClassMember

// This should not disable any warnings:
#pragma clang diagnostic ignored "-Wunused-local-typedef"
