// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wretained-language-linkage -DW_RETAINED_LANGUAGE_LINKAGE  %s
extern "C" {
  extern "C" void f(int);
}

extern "C++" {
  extern "C++" int& g(int);
  float& g();
}
double& g(double);

void test(int x, double d) {
  f(x);
  float &f1 = g();
  int& i1 = g(x);
  double& d1 = g(d);
}

extern "C" int foo;
extern "C" int foo;

extern "C" const int bar;
extern "C" int const bar;

// <rdar://problem/6895431>
extern "C" struct bar d;
extern struct bar e;

extern "C++" {
  namespace N0 {
    struct X0 {
      int foo(int x) { return x; }
    };
  }
}

// PR5430
namespace pr5430 {
  extern "C" void func(void);
}
using namespace pr5430;
extern "C" void pr5430::func(void) { }

// PR5405
int f2(char *)
{
        return 0;
}

extern "C"
{
    int f2(int)
    {
        return f2((char *)0);
    }
}

namespace PR5405 {
  int f2b(char *) {
    return 0;
  }

  extern "C" {
    int f2b(int) {
      return f2b((char *)0); // ok
    }
  }
}

// PR6991
extern "C" typedef int (*PutcFunc_t)(int);


// PR7859
extern "C" void pr7859_a(int) {} // expected-note {{previous definition}}
extern "C" void pr7859_a(int) {} // expected-error {{redefinition}}

extern "C" void pr7859_b() {} // expected-note {{previous definition}}
extern "C" void pr7859_b(int) {} // expected-error {{conflicting}}

extern "C" void pr7859_c(short) {} // expected-note {{previous definition}}
extern "C" void pr7859_c(int) {} // expected-error {{conflicting}}

// <rdar://problem/8318976>
extern "C" {
  struct s0 {
  private:
    s0();
    s0(const s0 &);
  };
}

//PR7754
extern "C++" template <class T> int pr7754(T param);

namespace N {
  int value;
}

extern "C++" using N::value;

// PR7076
extern "C" const char *Version_string = "2.9";

extern "C" {
  extern const char *Version_string2 = "2.9";
}

namespace PR9162 {
  extern "C" {
    typedef struct _ArtsSink ArtsSink;
    struct _ArtsSink {
      int sink;
    };
  }
  int arts_sink_get_type()
  {
    return sizeof(ArtsSink);
  }
}

namespace pr14958 {
  namespace js { extern int ObjectClass; }
  extern "C" {
    namespace js {}
  }
  int js::ObjectClass;
}

extern "C" void PR16167; // expected-error {{variable has incomplete type 'void'}}
extern void PR16167_0; // expected-error {{variable has incomplete type 'void'}}

// PR7927
enum T_7927 {
  E_7927
};

extern "C" void f_pr7927(int);

namespace {
  extern "C" void f_pr7927(int);

  void foo_pr7927() {
    f_pr7927(E_7927);
    f_pr7927(0);
    ::f_pr7927(E_7927);
    ::f_pr7927(0);
  }
}

void bar_pr7927() {
  f_pr7927(E_7927);
  f_pr7927(0);
  ::f_pr7927(E_7927);
  ::f_pr7927(0);
}

namespace PR17337 {
  extern "C++" {
    class Foo;
    extern "C" int bar3(Foo *y);
    class Foo {
      int x;
      friend int bar3(Foo *y);
#ifdef W_RETAINED_LANGUAGE_LINKAGE
// expected-note@-5 {{previous declaration is here}}
// expected-warning@-3 {{retaining previous language linkage}}
#endif
    };
    extern "C" int bar3(Foo *y) {
      return y->x;
    }
  }
}
