// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -fblocks -Wunused-label -verify=expected,obt %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -fblocks -Wunused-label -Woverflow-behavior-no-handler -verify=expected,obt,nohandler %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wunused-label -Wno-overflow-behavior-attribute-ignored -verify=expected,off %s

int no_effect(int a, int b) {
  return __builtin_overflow_goto(a * b, l); // obt-warning {{'__builtin_overflow_goto' has no effect because its operand performs no overflow-checked operation}}
l:
  return 0;
}

// The inner builtin takes every operation, so the outer one has no effect.
int nested(int __ob_trap a) {
  return __builtin_overflow_goto(__builtin_overflow_goto(a * 2, inner), outer); // obt-warning {{has no effect}}
inner:
  return -1;
outer:
  return -2;
}

// A local handler inside the operand takes over from the builtin's label.
int nested_local(int __ob_trap a) {
  return __builtin_overflow_goto(({ // obt-warning {{has no effect}}
    __label__ __overflow; // off-warning {{unused label '__overflow'}}
    int r = a + 1;
    if (0) {
    __overflow:
      r = 0;
    }
    r;
  }), l);
l:
  return 0;
}

void no_effect_block(int c) {
  __builtin_overflow_goto(({ if (c) c++; }), l); // obt-warning {{has no effect}}
l:;
}

// -Woverflow-behavior-no-handler reports each operation that traps.
int unhandled(int __ob_trap a) {
  unsigned u = a; // nohandler-warning {{overflow-checked operation has no overflow handler label '__overflow'; overflow will trap}}
  return a + u;   // nohandler-warning {{overflow-checked operation has no overflow handler label '__overflow'; overflow will trap}}
}

// Operations covered by the builtin are handled.
int handled_by_builtin(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}

// A block is a separate function and cannot reach its parent's handler.
int block(int __ob_trap a) {
  int (^b)(void) = ^{
    return a + 1; // obt-warning {{overflow in this block traps; it cannot jump to the enclosing function's overflow handler label '__overflow'}}
  };
  return b();
__overflow: // obt-note {{enclosing function's overflow handler label is here}} expected-warning {{unused label '__overflow'}}
  return -1;
}

// ... unless it defines its own.
int block_handler(int __ob_trap a) {
  int (^b)(void) = ^int(void) {
    return a + 1;
  __overflow: // off-warning {{unused label '__overflow'}}
    return -2;
  };
  return b();
}

// Checks in a variable length array size trap even though a handler exists.
int vla_size(int __ob_trap n) {
  int v[n + 1]; // nohandler-warning {{overflow-checked operation has no overflow handler label '__overflow'; overflow will trap}}
  v[0] = 0;
  return v[0];
__overflow: // expected-warning {{unused label '__overflow'}}
  return -1;
}
