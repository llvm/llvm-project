// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class X {};

void test() {
  X x;

  x.int; // expected-error{{expected unqualified-id}}
  x.~int(); // expected-error{{expected a class name}}
  x.operator; // expected-error{{expected a type}}
  x.operator typedef; // expected-error{{expected a type}} expected-error{{type name does not allow storage class}}
}

void test2() {
  X *x;

  x->int; // expected-error{{expected unqualified-id}}
  x->~int(); // expected-error{{expected a class name}}
  x->operator; // expected-error{{expected a type}}
  x->operator typedef; // expected-error{{expected a type}} expected-error{{type name does not allow storage class}}
}

// PR6327
namespace test3 {
  template <class A, class B> struct pair {};
  template <class _E> class initializer_list {};
  template <typename _Tp> pair<_Tp, _Tp> minmax(initializer_list<_Tp> __l) {};

  void test0() {
    pair<int, int> z = minmax({});
#if __cplusplus <= 199711L // C++03 or earlier modes
    // expected-error@-2 {{expected expression}}
#else
    // expected-error@-4 {{no matching function for call to 'minmax'}}
    // expected-note@-8 {{candidate template ignored: couldn't infer template argument '_Tp'}}
#endif
  }

  struct string {
    class iterator {};
  };

  void test1() {
    string s;
    string::iterator i = s.foo(); // expected-error {{no member named 'foo'}}
  }
}


// Make sure we don't crash.
namespace rdar11293995 {

struct Length {
  // FIXME: We try to annotate the template-id here during tentative parsing,
  // and fail, then try again during the actual parse. This results in the same
  // diagnostic being produced twice. :(
  explicit Length(PassRefPtr<CalculationValue>); // expected-error 2{{undeclared identifier 'CalculationValue'}}
};

struct LengthSize {
    Length m_width;
    Length m_height;
};

enum EFillSizeType { Contain, Cover, SizeLength, SizeNone };

struct FillSize {
    EFillSizeType type;
    LengthSize size;
};

class FillLayer {
public:
    void setSize(FillSize f) { m_sizeType = f.type;}
private:
    unsigned m_sizeType : 2;
};

}
