// RUN: %clang_cc1 -fsyntax-only -verify %s

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

  void test0() {
    pair<int, int> z = minmax({}); // expected-error {{expected expression}}
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
  explicit Length(PassRefPtr<CalculationValue>); // expected-error {{unknown type name}} \
                    expected-error {{expected ')'}} \
                    expected-note {{to match this '('}}
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
