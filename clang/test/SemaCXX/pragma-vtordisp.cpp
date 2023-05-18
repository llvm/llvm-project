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
#pragma vtordisp(pop) // stack should NOT be affected by reset.
                      // Now stack contains '1'.

#pragma vtordisp(      // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp(asdf) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp(,)    // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp       // expected-warning {{missing '(' after '#pragma vtordisp' - ignoring}}
#pragma vtordisp(3)    // expected-warning {{expected integer between 0 and 2 inclusive in '#pragma vtordisp' - ignored}}
#pragma vtordisp(), stuff // expected-warning {{extra tokens}}

struct C {
#pragma vtordisp()
  struct D : virtual A {
  };
};

struct E {
  virtual ~E();
  virtual void f();
};

#pragma vtordisp(pop) // After this stack should be empty.
#pragma vtordisp(pop) // expected-warning {{#pragma vtordisp(pop, ...) failed: stack empty}}

void g() {
  #pragma vtordisp(push, 2)
  struct F : virtual E {
    virtual ~F();
    virtual void f();
  };
}

#pragma vtordisp(pop) // OK because of local vtordisp(2) in g().

struct G {
  void f() {
    #pragma vtordisp(push, 2) // Method-local pragma - stack will be restored on exit.
  }
};

// Stack is restored on exit from G::f(), nothing to pop.
#pragma vtordisp(pop) // expected-warning {{#pragma vtordisp(pop, ...) failed: stack empty}}

int g2()
// FIXME: Our implementation based on token insertion makes it impossible for
// the pragma to appear everywhere we should support it.
// #pragma vtordisp()
{
  return 0;
}
