// RUN: %clang_cc1 -fsyntax-only -std=c++23 -fexperimental-overflow-behavior-types -verify %s

struct D {
  D(int);
  ~D();
};

// Skipping a constructor would run the destructor on an unconstructed object.
int destructor_init(int __ob_trap a) {
  D d(a + 1); // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
              // expected-note {{jump bypasses variable initialization}}
  return 0;
__overflow:
  return -1;
}

// Without a destructor, skipping the initialization runs no code.
struct P {
  int __ob_trap a, b;
};
int trivial_init(P p) {
  auto [x, y] = P{p.a + 1, p.b};
  return x;
__overflow:
  return -1;
}

int trivial_capture(P p) {
  auto l = [c = p.a + 2] { return c; };
  return l();
__overflow:
  return -1;
}

// As for goto, a jump may not bypass a later initialized declaration.
int later_declaration(int __ob_trap a) {
  a = a + 1; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int b = 2; // expected-note {{jump bypasses variable initialization}}
  return a + b;
__overflow:
  return -1;
}

// A handler outside the variable's scope is fine.
int destructor_init_outer(int __ob_trap a) {
  {
    D d(a + 1);
  }
  return 0;
__overflow:
  return -1;
}
