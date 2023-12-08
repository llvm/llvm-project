// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct global {
};

namespace PR10127 {
  struct outer {
    struct middle {
      struct inner {
        int func();
        int i;
      };
      struct inner2 {
      };
      struct inner3 {
      };
      int mfunc();
    };
    typedef int td_int;
  };

  struct str {
    operator decltype(outer::middle::inner()) ();
    operator decltype(outer::middle())::inner2 ();
    operator decltype(outer())::middle::inner3 ();
    str(int (decltype(outer::middle::inner())::*n)(),
             int (decltype(outer::middle())::inner::*o)(),
             int (decltype(outer())::middle::inner::*p)());
  };

  decltype(outer::middle::inner()) a;
  void scope() {
    a.decltype(outer::middle())::mfunc(); // expected-error{{'PR10127::outer::middle::mfunc' is not a member of class 'decltype(outer::middle::inner())'}}
    a.decltype(outer::middle::inner())::func();
    a.decltype(outer::middle())::inner::func();
    a.decltype(outer())::middle::inner::func();

    a.decltype(outer())::middle::inner::~inner();

    decltype(outer())::middle::inner().func();
  }
  decltype(outer::middle())::inner b;
  decltype(outer())::middle::inner c;
  decltype(outer())::fail d; // expected-error{{no type named 'fail' in 'PR10127::outer'}}
  decltype(outer())::fail::inner e; // expected-error{{no member named 'fail' in 'PR10127::outer'}}
  decltype()::fail f; // expected-error{{expected expression}}
  decltype()::middle::fail g; // expected-error{{expected expression}}
  
  decltype(int()) h;
  decltype(int())::PR10127::outer i; // expected-error{{'decltype(int())' (aka 'int') is not a class, namespace, or enumeration}}
  decltype(int())::global j; // expected-error{{'decltype(int())' (aka 'int') is not a class, namespace, or enumeration}}
  
  outer::middle k = decltype(outer())::middle();
  outer::middle::inner l = decltype(outer())::middle::inner();

  template<typename T>
  struct templ {
    typename decltype(T())::middle::inner x; // expected-error{{type 'decltype(int())' (aka 'int') cannot be used prior to '::' because it has no members}}
  };

  template class templ<int>; // expected-note{{in instantiation of template class 'PR10127::templ<int>' requested here}}
  template class templ<outer>;

  enum class foo {
    bar,
    baz
  };
  
  foo m = decltype(foo::bar)::baz;

  enum E {};
  enum H {};
  struct bar {
    enum E : decltype(outer())::td_int(4); // expected-error{{anonymous bit-field}}
    enum F : decltype(outer())::td_int;
    enum G : decltype; // expected-error{{expected '(' after 'decltype'}}
    enum H : 4; // expected-error {{anonymous bit-field}}
  };
}
