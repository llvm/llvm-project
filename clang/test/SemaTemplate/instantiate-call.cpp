// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N1 {
  struct X0 { };

  int& f0(X0);
}

namespace N2 {
  char& f0(char);

  template<typename T, typename Result>
  struct call_f0 {
    void test_f0(T t) {
      Result result = f0(t);
    }
  };
}

template struct N2::call_f0<int, char&>;
template struct N2::call_f0<N1::X0, int&>;

namespace N3 {
  template<typename T, typename Result>
  struct call_f0 {
    void test_f0(T t) {
      Result &result = f0(t); // expected-error {{undeclared identifier}} \
                                 expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
    }
  };
}

template struct N3::call_f0<int, char&>; // expected-note{{instantiation}}
template struct N3::call_f0<N1::X0, int&>;

short& f0(char); // expected-note {{should be declared prior to the call site}}
namespace N4 {
  template<typename T, typename Result>
  struct call_f0 {
    void test_f0(T t) {
      Result &result = f0(t);
    }
  };
}

template struct N4::call_f0<int, short&>;
template struct N4::call_f0<N1::X0, int&>;
template struct N3::call_f0<int, short&>; // expected-note{{instantiation}}

// FIXME: test overloaded function call operators, calls to member
// functions, etc.
