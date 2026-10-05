// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s --check-prefix=NEG

// A failed input must not take earlier declarations down with it, and must not
// leave anything of its own behind for a later input to trip over.

extern "C" int printf(const char *, ...);

namespace N { struct S { int v; }; void foo() { printf("foo\n"); } }

namespace N { void bar() { printf("bar\n" } }
// CHECK-DAG: error: expected ')'

// Everything N held before the failed input is still reachable.
N::foo();
// CHECK-DAG: foo
N::S s; s.v = 7; printf("s.v = %d\n", s.v);
// CHECK-DAG: s.v = 7

// N is still open for business, and bar is free to be defined properly.
namespace N { void bar() { printf("bar\n"); } }
N::bar();
// CHECK-DAG: bar
// NEG-NOT: error: call to 'bar' is ambiguous

namespace N { void baz() { printf("baz\n"); } }
N::baz();
// CHECK-DAG: baz

// A name that only ever existed in a failed input stays gone.
namespace M { int m = undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
int probe = M::m;
// CHECK-DAG: error: use of undeclared identifier 'M'

// A class survives a failed redefinition, and the failed definition does not
// become the one everybody sees.
struct T;
struct T { int a; }; int e1 = undeclared_thing;
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
T *tp = nullptr; printf("T reachable %d\n", tp == nullptr);
// CHECK-DAG: T reachable 1
struct T { int a; int b; };
printf("sizeof(T) = %d\n", (int)sizeof(T));
// CHECK-DAG: sizeof(T) = 

enum E : int;
enum E : int { A = 1 }; int e2 = undeclared_thing;
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
enum E : int { A = 1, B = 2 };
printf("B = %d\n", (int)B);
// CHECK-DAG: B = 2

// Namespace alias and using declaration.
namespace Deep { int v = 11; void g() { printf("Deep::g\n"); } }
namespace Al = Deep;
namespace Al = Deep; int e8 = undeclared_thing;
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
printf("Al::v = %d\n", Al::v);
// CHECK-DAG: Al::v = 11

using Deep::g;
using Deep::g; int e9 = undeclared_thing;
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
g();
// CHECK-DAG: Deep::g

// A member of a re-opened namespace is a redeclaration on its own.
namespace ns { class Foo; }
namespace ns { class Foo { public: int v; }; int e10 = undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
ns::Foo *fp = nullptr; printf("ns::Foo reachable %d\n", fp == nullptr);
// CHECK-DAG: ns::Foo reachable 1
namespace ns { class Foo { public: int v; int w; }; }
ns::Foo foo; foo.v = 1; foo.w = 2; printf("foo = %d %d\n", foo.v, foo.w);
// CHECK-DAG: foo = 1 2

namespace ns { void h(); }
namespace ns { void h() { printf("h discarded\n"); } int e11 = undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
namespace ns { void h() { printf("h kept\n"); } }
ns::h();
// CHECK-DAG: h kept
// NEG-NOT: {{^}}h discarded

// The same, one namespace deeper: the inner namespace is itself a member of
// the outer one.
namespace outer { namespace inner { class Bar; } }
namespace outer { namespace inner { class Bar { public: int v; }; } int e12 = undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
outer::inner::Bar *bp = nullptr; printf("outer::inner::Bar reachable %d\n", bp == nullptr);
// CHECK-DAG: outer::inner::Bar reachable 1

// Anonymous namespace
namespace { int anon_v = 11; } int e13 = undeclared_thing;
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
namespace { int anon_v = 22; }
printf("anon_v = %d\n", anon_v);
// CHECK-DAG: anon_v = 22

%quit
