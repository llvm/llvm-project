// RUN: %clang_cc1 -fsyntax-only -verify -fincremental-extensions -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -DMS -fms-extensions -verify -fincremental-extensions -std=c++20 %s

// expected-no-diagnostics

extern "C" int printf(const char*,...);

// Decls which are hard to disambiguate

// Templates
namespace ns1 { template<typename T> void tmplt(T &) {}}
int arg_tmplt = 12; ns1::tmplt(arg_tmplt);

// ParseStatementOrDeclaration returns multiple statements.
#ifdef MS
int g_bFlag = 1;
__if_exists(::g_bFlag) {
  printf("Entering __if_exists\n");
  printf("g_bFlag = %d\n", g_bFlag);
}
#endif // MS

// Operators.
struct S1 { operator int(); };
S1::operator int() { return 0; }

// Dtors
using I = int;
I x = 10;
x.I::~I();
x = 20;

struct Dtor1 {~Dtor1();};
Dtor1::~Dtor1() { printf("Dtor1\n"); }
Dtor1 d1;

struct Dtor2 { ~Dtor2(); };
::Dtor2::~Dtor2() { printf("Dtor2\n"); }
Dtor2 d2;

struct ANestedDtor { struct A1 { struct A2 { ~A2(); }; }; };
ANestedDtor::A1::A2::~A2() { printf("Dtor A::A1::A2::~A2\n"); }

// Ctors

// Deduction guide
template<typename T> struct A { A(); A(T); };
A() -> A<int>;

struct S2 { S2(); };
S2::S2() = default;

namespace N { struct S { S(); }; }
N::S::S() { printf("N::S::S()\n"); }
N::S s;

namespace Ns {namespace Ns { void Ns(); void Fs();}}
void Ns::Ns::Ns() { printf("void Ns::Ns::Ns()\n"); }
void Ns::Ns::Fs() {}

Ns::Ns::Fs();
Ns::Ns::Ns();

struct Attrs1 { Attrs1(); };
Attrs1::Attrs1() __attribute((pure)) = default;

struct Attrs2 { Attrs2(); };
__attribute((pure)) Attrs2::Attrs2() = default;

// Extra semicolon
namespace N {};
