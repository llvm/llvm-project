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

// Kinds reached only through the generated switch, not by any hand-written
// list: a namespace alias and a using declaration.
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

%quit
