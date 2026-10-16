// RUN: cat %s | clang-repl | FileCheck %s

// Undoing the out-of-line definition of a member restores its previous
// declaration in the lookup table of its class or namespace, not in the
// enclosing context of the definition.

struct K { static int s; };
int K::s = 4;
%undo
int K::s = 5;
K::s
// CHECK: (int) 5

struct C { int m(); };
int C::m() { return 1; }
%undo
int C::m() { return 2; }
C().m()
// CHECK-NEXT: (int) 2

namespace M { int f(); extern int v; }
int M::f() { return 1; }
int M::v = 1;
%undo
%undo
int M::f() { return 3; }
int M::v = 4;
M::f() + M::v
// CHECK-NEXT: (int) 7

// The members are not visible in the enclosing context.
int s = 8, m = 9, f = 10, v = 11;
s + m + f + v
// CHECK-NEXT: (int) 38

%quit
