// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// Test the X can be overloaded inside the struct.
typedef int X; 
struct Y { short X; };

// Variable shadows type, PR3872

typedef struct foo { int x; } foo;
void test() {
   foo *foo;
   foo->x = 0;
}

