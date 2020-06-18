// RUN: %clang_cc1 -verify -std=c++11 -fms-extensions %s

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

namespace NS {
  template<typename T> struct Foo {};
}
struct MissingSemiThenTemplate1 {} // expected-error {{expected ';' after struct}}
NS::Foo<int> missingSemiBeforeFunctionReturningTemplateId1();

using NS::Foo;
struct MissingSemiThenTemplate2 {} // expected-error {{expected ';' after struct}}
Foo<int> missingSemiBeforeFunctionReturningTemplateId2();

namespace PR17084 {
enum class EnumID {};
template <typename> struct TempID;
template <> struct TempID<BadType> // expected-error{{use of undeclared identifier 'BadType'}}
  : BadType, // expected-error {{expected class name}}
    EnumID::Garbage // expected-error {{expected class name}}
  ; // expected-error@-1 {{expected '{' after base class list}}
}

namespace pr15133 {
  namespace ns {
    const int V1 = 1;   // expected-note {{declared here}}
  }
  struct C1 {
    enum E1 { V2 = 2 }; // expected-note {{declared here}}
    static const int V3 = 3; // expected-note {{declared here}}
  };
  enum E2 {
    V4 = 4,   // expected-note {{declared here}}
    V6        // expected-note {{declared here}}
  };
  enum class EC3 { V0 = 0, V5 = 5 }; // expected-note {{declared here}}
  void func_3();

  void func_1(int x) {
    switch(x) {
    case 0: break;
    case ns::V1:: break; // expected-error{{'V1' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    case C1::V2:: break; // expected-error{{'V2' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    case C1::V3:: break; // expected-error{{'V3' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    case V4:: break; // expected-error{{'V4' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    case V6:: func_3();   // expected-error{{'V6' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    }
  }
  void func_2(EC3 x) {
    switch(x) {
    case EC3::V0:  break;
    case EC3::V5:: break; // expected-error{{'V5' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
    }
  }

  template<class T> struct TS1 {
    typedef int A;
  };
  template<class T> void func(int x) {
    switch(x) {
    case TS1<T>::A:: break;  // expected-error{{expected unqualified-id}}
    }
  };
  void mainf() {
    func<int>(1);
  }

  struct S {
    static int n;  // expected-note{{declared here}}
    int nn;        // expected-note 2 {{declared here}}
  };

  int func_3(int x) {
    return x ? S::n :: 0;  // expected-error{{'n' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
  }
  int func_4(int x, S &s) {
    return x ? s.nn :: x;  // expected-error{{'nn' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
  }
  int func_5(int x, S &s) {
    return x ? s.nn :: S::n;  // expected-error{{'nn' cannot appear before '::' because it is not a class, namespace, or enumeration; did you mean ':'?}}
  }

  struct S2 {
    struct S3;
  };

  struct S2 :: S3 :: public S2 {  // expected-error{{'public' cannot be a part of nested name specifier; did you mean ':'?}}
  };
}

namespace InvalidEmptyNames {
// These shouldn't crash, the diagnostics aren't important.
struct ::, struct ::; // expected-error 2 {{expected identifier}} expected-error 2 {{declaration of anonymous struct must be a definition}} expected-warning {{declaration does not declare anything}}
enum ::, enum ::; // expected-error 2 {{expected identifier}} expected-warning {{declaration does not declare anything}}
struct ::__super, struct ::__super; // expected-error 2 {{expected identifier}} expected-error 2 {{expected '::' after '__super'}}
struct ::template foo, struct ::template bar; // expected-error 2 {{expected identifier}} expected-error 2 {{declaration of anonymous struct must be a definition}} expected-warning {{declaration does not declare anything}}
struct ::foo struct::; // expected-error {{no struct named 'foo' in the global namespace}} expected-error {{expected identifier}} expected-error {{declaration of anonymous struct must be a definition}}
class :: : {} a;  // expected-error {{expected identifier}} expected-error {{expected class name}}
}
