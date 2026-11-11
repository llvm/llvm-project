// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -xc | FileCheck %s

// In C, the tags and enumerators declared in a structure or union are visible
// in the enclosing scope. Undoing the structure also undoes them.
//
// The first value printing looks up the value printing functions in the
// translation unit. That builds its lookup table, which then misses B, E and
// Red because they were declared inside of A.

struct A { struct B { int x; } b; enum E { Red } e; enum { Green } g; };
1
%undo
%undo
// CHECK: (int) 1
struct B { double y; };
enum E { Blue = 3 };
int Red = 5;
int Green = 6;
(struct B){1.5}.y
// CHECK-NEXT: (double) 1.5000000
Blue
// CHECK-NEXT: (int) 3
Red
// CHECK-NEXT: (int) 5
Green
// CHECK-NEXT: (int) 6

%quit
