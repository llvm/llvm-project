// RUN: %clang_cc1 -fsyntax-only -verify %s 
namespace Ns {
  int f(); // expected-note{{previous declaration is here}}

  enum E {
    Enumerator
  };
}
namespace Ns {
  double f(); // expected-error{{functions that differ only in their return type cannot be overloaded}}

  int x = Enumerator;
}

namespace Ns2 {
  float f();
}

int y = Ns::Enumerator;

namespace Ns2 {
  float f(int); // expected-note{{previous declaration is here}}
}

namespace Ns2 {
  double f(int); // expected-error{{functions that differ only in their return type cannot be overloaded}}
}

namespace N {
  int& f1();
}

namespace N {
  struct f1 {
    static int member;

    typedef int type;

    void foo(type);
  };

  void test_f1() {
    int &i1 = f1();
  }
}

void N::f1::foo(int i) { 
  f1::member = i; 
  f1::type &ir = i;
}

namespace N {
  float& f1(int x) {
    N::f1::type& i1 = x;
    f1::type& i2 = x;
  }

  struct f2 {
    static int member;
  };
  void f2();
}

int i1 = N::f1::member;
typedef struct N::f1 type1;
int i2 = N::f2::member;
typedef struct N::f2 type2;

void test_f1(int i) {
  int &v1 = N::f1();
  float &v2 = N::f1(i);
  int v3 = ::i1;
  int v4 = N::f1::member;
}

typedef int f2_type;
namespace a {
  typedef int f2_type(int, int);

  void test_f2() {
    ::f2_type(1, 2); // expected-error {{excess elements in scalar initializer}}
  }
}

// PR clang/3291
namespace a {  
  namespace a {   // A1
    namespace a { // A2
      int i; // expected-note{{'::a::a::a::i' declared here}}
    }
  }
}

void test_a() {
  a::a::i = 3; // expected-error{{no member named 'i' in namespace 'a::a'; did you mean '::a::a::a::i'?}}
  a::a::a::i = 4;
  a::a::j = 3; // expected-error-re{{no member named 'j' in namespace 'a::a'$}}
}
  
struct Undef { // expected-note{{definition of 'Undef' is not complete until the closing '}'}}
  typedef int type;

  Undef::type member;

  static int size = sizeof(Undef); // expected-error{{invalid application of 'sizeof' to an incomplete type 'Undef'}}

  int f();
};

int Undef::f() {
  return sizeof(Undef);
}

// PR clang/5667
namespace test1 {
  template <typename T> struct is_class {
    enum { value = 0 };
  };

  template <typename T> class ClassChecker {
    bool isClass() {
      return is_class<T>::value;
    }
  };

  template class ClassChecker<int>;  
}

namespace PR6830 {
  namespace foo {

    class X {
    public:
      X() {}
    };

  }  // namespace foo

  class Z {
  public:
    explicit Z(const foo::X& x) {}

    void Work() {}
  };

  void Test() {
    Z(foo::X()).Work();
  }
}

namespace pr12339 {
  extern "C" void i;
  pr12339::FOO  // expected-error{{no type named 'FOO' in namespace 'pr12339'}}
}  // expected-error{{expected unqualified-id}}
