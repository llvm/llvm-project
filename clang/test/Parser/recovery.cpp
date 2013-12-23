// RUN: %clang_cc1 -verify -std=c++11 %s

8gi///===--- recovery.cpp ---===// // expected-error {{unqualified-id}}
namespace Std { // expected-note {{here}}
  typedef int Important;
}

/ redeclare as an inline namespace // expected-error {{unqualified-id}}
inline namespace Std { // expected-error {{cannot be reopened as inline}}
  Important n;
} / end namespace Std // expected-error {{unqualified-id}}
int x;
Std::Important y;

extenr "C" { // expected-error {{did you mean 'extern'}}
  void f();
}
void g() {
  z = 1; // expected-error {{undeclared}}
  f();
}

struct S {
  int a, b, c;
  S();
  int x // expected-error {{expected ';'}}
  friend void f()
};
8S::S() : a{ 5 }, b{ 6 }, c{ 2 } { // expected-error {{unqualified-id}}
  return;
}
int k;
int l = k // expected-error {{expected ';'}}
constexpr int foo();

5int m = { l }, n = m; // expected-error {{unqualified-id}}

namespace MissingBrace {
  struct S { // expected-error {{missing '}' at end of definition of 'MissingBrace::S'}}
    int f();
  // };

  namespace N { int g(); } // expected-note {{still within definition of 'MissingBrace::S' here}}

  int k1 = S().h(); // expected-error {{no member named 'h' in 'MissingBrace::S'}}
  int k2 = S().f() + N::g();

  template<typename T> struct PR17949 { // expected-error {{missing '}' at end of definition of 'MissingBrace::PR17949'}}

  namespace X { // expected-note {{still within definition of 'MissingBrace::PR17949' here}}
  }
}

namespace N {
  int
} // expected-error {{unqualified-id}}

strcut Uuuu { // expected-error {{did you mean 'struct'}} \
              // expected-note {{'Uuuu' declared here}}
} *u[3];
uuuu v; // expected-error {{did you mean 'Uuuu'}}

struct Redefined { // expected-note {{previous}}
  Redefined() {}
};
struct Redefined { // expected-error {{redefinition}}
  Redefined() {}
};

struct MissingSemi5;
namespace N {
  typedef int afterMissingSemi4;
  extern MissingSemi5 afterMissingSemi5;
}

struct MissingSemi1 {} // expected-error {{expected ';' after struct}}
static int afterMissingSemi1();

class MissingSemi2 {} // expected-error {{expected ';' after class}}
MissingSemi1 *afterMissingSemi2;

enum MissingSemi3 {} // expected-error {{expected ';' after enum}}
::MissingSemi1 afterMissingSemi3;

extern N::afterMissingSemi4 afterMissingSemi4b;
union MissingSemi4 { MissingSemi4(int); } // expected-error {{expected ';' after union}}
N::afterMissingSemi4 (afterMissingSemi4b);

int afterMissingSemi5b;
struct MissingSemi5 { MissingSemi5(int); } // ok, no missing ';' here
N::afterMissingSemi5 (afterMissingSemi5b);

template<typename T> struct MissingSemiT {
} // expected-error {{expected ';' after struct}}
MissingSemiT<int> msi;

struct MissingSemiInStruct {
  struct Inner1 {} // expected-error {{expected ';' after struct}}
  static MissingSemi5 ms1;

  struct Inner2 {} // ok, no missing ';' here
  static MissingSemi1;

  struct Inner3 {} // expected-error {{expected ';' after struct}}
  static MissingSemi5 *p;
};

void MissingSemiInFunction() {
  struct Inner1 {} // expected-error {{expected ';' after struct}}
  if (true) {}

  // FIXME: It would be nice to at least warn on this.
  struct Inner2 { Inner2(int); } // ok, no missing ';' here
  k = l;

  struct Inner3 {} // expected-error {{expected ';' after struct}}
  Inner1 i1;

  struct Inner4 {} // ok, no missing ';' here
  Inner5;
}

namespace PR17084 {
enum class EnumID {};
template <typename> struct TempID;
template <> struct TempID<BadType> : BadType, EnumID::Garbage; // expected-error{{use of undeclared identifier 'BadType'}}
}
