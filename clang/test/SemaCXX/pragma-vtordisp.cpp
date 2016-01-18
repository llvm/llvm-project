// RUN: %clang_cc1 -std=c++11 -fms-extensions -fms-compatibility -fsyntax-only -triple=i386-pc-win32 -verify %s

struct A { int a; };

#pragma vtordisp(pop) // expected-warning {{#pragma vtordisp(pop, ...) failed: stack empty}}
#pragma vtordisp(push, 0)
#pragma vtordisp(push, 1)
#pragma vtordisp(push, 2)
struct B : virtual A { int b; };
#pragma vtordisp(pop)
#pragma vtordisp(pop)
#pragma vtordisp(pop)
#pragma vtordisp(pop) // expected-warning {{#pragma vtordisp(pop, ...) failed: stack empty}}

#pragma vtordisp(push, 3) // expected-warning {{expected integer between 0 and 2 inclusive in '#pragma vtordisp' - ignored}}
#pragma vtordisp()

#define ONE 1
#pragma vtordisp(push, ONE)
#define TWO 1
#pragma vtordisp(push, TWO)

// Test a reset.
#pragma vtordisp()
#pragma vtordisp(pop) // expected-warning {{#pragma vtordisp(pop, ...) failed: stack empty}}

#pragma vtordisp(      // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp(asdf) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp(,)    // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp       // expected-warning {{missing '(' after '#pragma vtordisp' - ignoring}}
#pragma vtordisp(3)    // expected-warning {{expected integer between 0 and 2 inclusive in '#pragma vtordisp' - ignored}}
#pragma vtordisp(), stuff // expected-warning {{extra tokens}}

struct C {
// FIXME: Our implementation based on token insertion makes it impossible for
// the pragma to appear everywhere we should support it.
//#pragma vtordisp()
  struct D : virtual A {
  };
};
