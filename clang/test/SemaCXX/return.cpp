// RUN: %clang_cc1 %s -std=c++11 -fcxx-exceptions -fexceptions -fsyntax-only -Wignored-qualifiers -verify

int test1() {
  throw;
}

// PR5071
template<typename T> T f() { }

template<typename T>
void g(T t) {
  return t * 2; // okay
}

template<typename T>
T h() {
  return 17;
}

// Don't warn on cv-qualified class return types, only scalar return types.
namespace ignored_quals {
struct S {};
const S class_c();
const volatile S class_cv();

const int scalar_c(); // expected-warning{{'const' type qualifier on return type has no effect}}
int const scalar_c2(); // expected-warning{{'const' type qualifier on return type has no effect}}

const
char*
const // expected-warning{{'const' type qualifier on return type has no effect}}
f();

char
const*
const // expected-warning{{'const' type qualifier on return type has no effect}}
g();

char* const h(); // expected-warning{{'const' type qualifier on return type has no effect}}
char* volatile i(); // expected-warning{{'volatile' type qualifier on return type has no effect}}

char*
volatile // expected-warning{{'const volatile' type qualifiers on return type have no effect}}
const
j();

const volatile int scalar_cv(); // expected-warning{{'const volatile' type qualifiers on return type have no effect}}

// FIXME: Maintain enough information that we can point the diagnostic at the 'volatile' keyword.
const
int S::*
volatile
mixed_ret(); // expected-warning {{'volatile' type qualifier on return type has no effect}}

const int volatile // expected-warning {{'const volatile' type qualifiers on return type have no effect}}
    (((parens())));

_Atomic(int) atomic();

_Atomic // expected-warning {{'_Atomic' type qualifier on return type has no effect}}
    int
    atomic();

auto
    trailing_return_type() -> // expected-warning {{'const' type qualifier on return type has no effect}}
    const int;

const int ret_array()[4]; // expected-error {{cannot return array}}
}

namespace PR9328 {
  typedef char *PCHAR;
  class Test 
  {
    const PCHAR GetName() { return 0; } // expected-warning{{'const' type qualifier on return type has no effect}}
  };
}

class foo  {
  operator const int ();
  operator int * const ();
};

namespace PR10057 {
  struct S {
    ~S();
  };

  template <class VarType>
  void Test(const VarType& value) {
    return S() = value;
  }
}

namespace return_has_expr {
  struct S {
    S() {
      return 42; // expected-error {{constructor 'S' should not return a value}}
    }
    ~S() {
      return 42; // expected-error {{destructor '~S' should not return a value}}
    }
  };
}

// rdar://15366494
// pr17759
namespace ctor_returns_void {
  void f() {}
  struct S { 
    S() { return f(); }; // expected-error {{constructor 'S' must not return void expression}}
    ~S() { return f(); } // expected-error {{destructor '~S' must not return void expression}}
  };
}

void cxx_unresolved_expr() {
  // The use of an undeclared variable tricks clang into building a
  // CXXUnresolvedConstructExpr, and the missing ')' gives it an invalid source
  // location for its rparen.  Check that emitting a diag on the range of the
  // expr doesn't assert.
  return int(undeclared, 4; // expected-error {{expected ')'}} expected-note{{to match this '('}} expected-error {{use of undeclared identifier 'undeclared'}}
}
