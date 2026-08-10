// RUN: %clang_cc1 -fsyntax-only -verify -fincremental-extensions -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -DMS -fms-extensions -verify -fincremental-extensions -std=c++20 %s

extern "C" int printf(const char*,...);

// Decls which are hard to disambiguate

// Templates
namespace ns1 { template<typename T> void tmplt(T &) {}}
int arg_tmplt = 12; ns1::tmplt(arg_tmplt);

namespace ns2 { template <typename T> struct S {}; }
namespace ns3 { struct A { public: using S = int; }; }
namespace ns3 { A::S f(A::S a); }

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

// Private typedefs / using declarations
class PrivateUsingMember { using T = int; T f(); };
PrivateUsingMember::T PrivateUsingMember::f() { return 0; }

class PrivateUsingVar { using T = int; static T i; };
PrivateUsingVar::T PrivateUsingVar::i = 42;

// The same with namespaces
namespace PrivateUsingNamespace { class Member { using T = int; T f(); }; }
PrivateUsingNamespace::Member::T PrivateUsingNamespace::Member::f() { return 0; }

namespace PrivateUsingNamespace { class Var { using T = int; static T i; }; }
PrivateUsingNamespace::Var::T PrivateUsingNamespace::Var::i = 42;

// The same with friend declarations
class PrivateUsingFriendMember;
class PrivateUsingFriendVar;
class PrivateUsingFriend { friend class PrivateUsingFriendMember; friend class PrivateUsingFriendVar; using T = int; };
class PrivateUsingFriendMember { PrivateUsingFriend::T f(); };
PrivateUsingFriend::T PrivateUsingFriendMember::f() { return 0; }

class PrivateUsingFriendVar { static PrivateUsingFriend::T i; };
PrivateUsingFriend::T PrivateUsingFriendVar::i = 42;

// The following should still diagnose (inspired by PR13642)
class PR13642 { class Inner { public: static int i; }; };
// expected-note@-1 {{implicitly declared private here}}
PR13642::Inner::i = 5;
// expected-error@-1 {{'Inner' is a private member of 'PR13642'}}

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
Attrs1::Attrs1() __attribute((noreturn)) = default;

struct Attrs2 { Attrs2(); };
__attribute((noreturn)) Attrs2::Attrs2() = default;

// Extra semicolon
namespace N {};

// Test C keywords supported in all language modes.
// https://clang.llvm.org/docs/LanguageExtensions.html#c-keywords-supported-in-all-language-modes

_Alignas(16) int aligned_var;
int align = _Alignof(double);
_Atomic int atomic_var = 0;
_Complex double complex_val = 1.0 + 2.0i;
_Float16 f = 1.5; // expected-error 0-1{{_Float16 is not supported on this target}}
_Thread_local int counter = 0;
_Static_assert(sizeof(int) == 4, "int must be 4 bytes");
_Imaginary float i = 2.0f; // expected-error {{imaginary types are not supported}}
_Noreturn void noreturn_func() { while (true) {} }