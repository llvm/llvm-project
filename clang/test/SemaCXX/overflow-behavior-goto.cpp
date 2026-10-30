// RUN: %clang_cc1 -fsyntax-only -std=c++23 -fcxx-exceptions -fexperimental-overflow-behavior-types -Wunused-label -verify %s

// A dependent operand is rebuilt per instantiation.
template <typename T> T dep(T a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}
template int dep(int);
template int __ob_trap dep(int __ob_trap);

// Instantiation rebinds the label.
template <typename T> int inst(T a) {
  T r = __builtin_overflow_goto(a * 2, too_big);
  int v[1];
  return r + v[0];
too_big:
  return -1;
}
template int inst(int __ob_trap);

// Capture initializers belong to the enclosing function.
int capture_init(int __ob_trap a) {
  auto f = [b = a + 1] { return 0; };
  return f();
__overflow:
  return -1;
}

// The result is an lvalue when the operand is.
int lvalue(int __ob_trap a) {
  __builtin_overflow_goto(a += 1, l) = 3;
  return a;
l:
  return 0;
}

// The builtin is transparent to constant evaluation.
constexpr int ce(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}
static_assert(ce(1) == 2);
constexpr int bad = ce(0x7fffffff); // expected-error {{constexpr variable 'bad' must be initialized by a constant expression}} \
                                    // expected-note@-5 {{value 2147483648 is outside the range of representable values}} \
                                    // expected-note {{in call to}}

// Checks in these contexts trap, and the builtin may not branch out of them.
struct T {
  T(int);
};
struct S {
  int __ob_trap x;
  S(int __ob_trap a) : x(__builtin_overflow_goto(a + 1, l)) { // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                                              // expected-note {{jump leaves a constructor initializer}}
    return;
  l:
    x = 0;
  }
};

T *new_init(int __ob_trap a) {
  return new T(__builtin_overflow_goto(a + 1, l)); // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                                    // expected-note {{jump leaves the initializer of a new expression}}
l:
  return nullptr;
}

void throw_operand(int __ob_trap a) {
  throw T(__builtin_overflow_goto(a + 1, l)); // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                               // expected-note {{jump leaves the operand of a throw expression}}
l:
  return;
}

int static_init(int __ob_trap a) {
  static int s = __builtin_overflow_goto(a + 1, l); // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                                     // expected-note {{jump leaves the initializer of a static local variable}}
  return s;
l:
  return -1;
}

// A label within the context is fine.
T *new_init_inner(int __ob_trap a) {
  return new T(({
    int r = __builtin_overflow_goto(a + 1, l);
    if (0) {
    l:
      r = 0;
    }
    r;
  }));
}
