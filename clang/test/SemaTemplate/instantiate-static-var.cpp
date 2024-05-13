// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T, T Divisor>
class X {
public:
  static const T value = 10 / Divisor; // expected-error{{in-class initializer for static data member is not a constant expression}}
};

int array1[X<int, 2>::value == 5? 1 : -1];
X<int, 0> xi0; // expected-note{{in instantiation of template class 'X<int, 0>' requested here}}


template<typename T>
class Y {
  static const T value = 0; 
#if __cplusplus <= 199711L
// expected-warning@-2 {{in-class initializer for static data member of type 'const float' is a GNU extension}}
#else
// expected-error@-4 {{in-class initializer for static data member of type 'const float' requires 'constexpr' specifier}}
// expected-note@-5 {{add 'constexpr'}}
#endif
};

Y<float> fy; // expected-note{{in instantiation of template class 'Y<float>' requested here}}


// out-of-line static member variables

template<typename T>
struct Z {
  static T value;
};

template<typename T>
T Z<T>::value; // expected-error{{no matching constructor}}

struct DefCon {};

struct NoDefCon { 
  NoDefCon(const NoDefCon&); // expected-note{{candidate constructor}}
};

void test() {
  DefCon &DC = Z<DefCon>::value;
  NoDefCon &NDC = Z<NoDefCon>::value; // expected-note{{instantiation}}
}

// PR5609
struct X1 {
  ~X1();  // The errors won't be triggered without this dtor.
};

template <typename T>
struct Y1 {
  static char Helper(T);
  static const int value = sizeof(Helper(T()));
};

struct X2 {
  virtual ~X2();
};

namespace std {
  class type_info { };
}

template <typename T>
struct Y2 {
  static T &Helper();
  static const int value = sizeof(typeid(Helper()));
};

template <int>
struct Z1 {};

void Test() {
  Z1<Y1<X1>::value> x;
  int y[Y1<X1>::value];
  Z1<Y2<X2>::value> x2;
  int y2[Y2<X2>::value];
}

// PR5672
template <int n>
struct X3 {};

class Y3 {
 public:
  ~Y3();  // The error isn't triggered without this dtor.

  void Foo(X3<1>);
};

template <typename T>
struct SizeOf {
  static const int value = sizeof(T);
};

void MyTest3() {
   Y3().Foo(X3<SizeOf<char>::value>());
}

namespace PR6449 {
  template<typename T>    
  struct X0  {
    static const bool var = false;
  };

  template<typename T>
  const bool X0<T>::var;

  template<typename T>
  struct X1 : public X0<T> {
    static const bool var = false;
  };

  template<typename T>      
  const bool X1<T>::var;

  template class X0<char>;
  template class X1<char>;

}

typedef char MyString[100];
template <typename T>
struct StaticVarWithTypedefString {
  static MyString str;
};
template <typename T>
MyString StaticVarWithTypedefString<T>::str = "";

void testStaticVarWithTypedefString() {
  (void)StaticVarWithTypedefString<int>::str;
}

namespace ArrayBound {
#if __cplusplus >= 201103L
  template<typename T> void make_unique(T &&);

  template<typename> struct Foo {
    static constexpr char kMessage[] = "abc";
    static void f() { make_unique(kMessage); }
    static void g1() { const char (&ref)[4] = kMessage; } // OK
    // We can diagnose this prior to instantiation because kMessage is not type-dependent.
    static void g2() { const char (&ref)[5] = kMessage; } // expected-error {{could not bind}}
  };
  template void Foo<int>::f();
#endif

  template<typename> struct Bar {
    static const char kMessage[];
    // Here, kMessage is type-dependent, so we don't diagnose until
    // instantiation.
    static void g1() { const char (&ref)[4] = kMessage; } // expected-error {{could not bind to an lvalue of type 'const char[5]'}}
    static void g2() { const char (&ref)[5] = kMessage; } // expected-error {{could not bind to an lvalue of type 'const char[4]'}}
  };
  template<typename T> const char Bar<T>::kMessage[] = "foo";
  template void Bar<int>::g1();
  template void Bar<int>::g2(); // expected-note {{in instantiation of}}

  template<> const char Bar<char>::kMessage[] = "foox";
  template void Bar<char>::g1(); // expected-note {{in instantiation of}}
  template void Bar<char>::g2();
}
