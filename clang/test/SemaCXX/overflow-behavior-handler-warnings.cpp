// RUN: %clang_cc1 -fsyntax-only -std=c++23 -fexperimental-overflow-behavior-types -Wunused-label -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -fexperimental-overflow-behavior-types -Wunused-label -Woverflow-behavior-no-handler -verify=expected,nohandler %s

// A non-dependent operand is diagnosed once, in the template definition.
template <typename T> int nondep(int a, T) {
  return __builtin_overflow_goto(a + 1, l); // expected-warning {{'__builtin_overflow_goto' has no effect}}
l:
  return 0;
}
template int nondep(int, int);
template int nondep(int, long);

// A dependent operand is never diagnosed, even in instantiations where it
// performs no checked operation.
template <typename T> T dep(T a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}
template int dep(int);
template int __ob_trap dep(int __ob_trap);

// Lambdas cannot reach the enclosing function's handler.
int lambda(int __ob_trap a) {
  auto f = [&] { return a * 2; }; // expected-warning {{overflow in this lambda traps; it cannot jump to the enclosing function's overflow handler label '__overflow'}}
  auto g = [&] {
    int r = a * 3;
    return r;
  __overflow:
    return -1;
  };
  return f() + g();
__overflow: // expected-note {{enclosing function's overflow handler label is here}}
  return -1;
}

// Nor the label of an enclosing builtin.
int lambda_in_builtin(int __ob_trap a) {
  return __builtin_overflow_goto([&] { return a + 1; }(), l); // expected-warning {{has no effect}} expected-warning {{overflow in this lambda traps; it cannot jump to the enclosing function's overflow handler label 'l'}}
l: // expected-note {{enclosing function's overflow handler label is here}}
  return 0;
}

// A member function of a local class is a separate function: it uses its own
// handler and cannot reach the enclosing function's.
int local_class(int __ob_trap x) {
  struct S {
    int none(int __ob_trap a) {
      return a + 1; // nohandler-warning {{overflow-checked operation has no overflow handler label '__overflow'; overflow will trap}}
    }
    int own(int __ob_trap a) {
      return a * 2;
    __overflow:
      return -1;
    }
    int lambda(int __ob_trap a) {
      auto f = [&] { return a - 1; }; // expected-warning {{overflow in this lambda traps; it cannot jump to the enclosing function's overflow handler label '__overflow'}}
      return f();
    __overflow: // expected-note {{enclosing function's overflow handler label is here}} expected-warning {{unused label '__overflow'}}
      return -1;
    }
  };
  S s;
  return s.none(x) + s.own(x) + s.lambda(x);
__overflow: // expected-warning {{unused label '__overflow'}}
  return -1;
}

// Local classes in templates are diagnosed per instantiation.
template <typename T> int local_class_template(T x) {
  struct S { // nohandler-note {{in instantiation of member function 'local_class_template(int)::S::none' requested here}}
    int none(int __ob_trap a) {
      return a + 1; // nohandler-warning {{overflow-checked operation has no overflow handler label '__overflow'; overflow will trap}}
    }
  };
  return S().none(x);
}
template int local_class_template(int); // nohandler-note {{in instantiation of function template specialization 'local_class_template<int>' requested here}}
