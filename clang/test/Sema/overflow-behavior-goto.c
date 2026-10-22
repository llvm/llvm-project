// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -fblocks -Wunused-label -verify=expected,obt -DHAS_BUILTIN=1 %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wunused-label -Wno-overflow-behavior-attribute-ignored -verify=expected,off -DHAS_BUILTIN=0 %s

#if __has_builtin(__builtin_overflow_goto) != HAS_BUILTIN
#error unexpected __has_builtin result
#endif

void fr(int *);

// The builtin's label is used even when no default handler exists.
int redirect(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, too_big);
too_big:
  return -1;
}

// Operations outside the builtin still use the default handler.
int both(int __ob_trap a) {
  a = a * 2;
  a = __builtin_overflow_goto(a + 1, too_big);
  return a;
__overflow: // off-warning {{unused label '__overflow'}}
  return -1;
too_big:
  return -2;
}

int missing(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, nowhere); // expected-error {{use of undeclared label 'nowhere'}}
}

int not_integer(float f) {
  return __builtin_overflow_goto(f, l); // expected-error {{operand of '__builtin_overflow_goto' must have integer or void type ('float' invalid)}}
l:
  return 0;
}

// A void operand lets a statement expression guard a whole block.
void stmt_block(int __ob_trap a, int c) {
  __builtin_overflow_goto(({ a--; if (c) a += 2; }), l);
  __builtin_overflow_goto(({ int __attribute__((cleanup(fr))) x = 0; a += x; }), l);
l:;
}

int not_label(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, 3); // expected-error {{expected identifier}}
}

int file_scope = __builtin_overflow_goto(1, l); // expected-error {{'__builtin_overflow_goto' cannot be used outside of a function body}}

// The jump obeys goto scope rules.
void into_vla(int __ob_trap a, int n) {
  a = __builtin_overflow_goto(a + 1, l); // obt-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int v[n];                              // obt-note {{jump bypasses initialization of variable length array}}
l:
  ;
}

void into_cleanup(int __ob_trap a) {
  a = __builtin_overflow_goto(a + 1, l); // obt-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int z __attribute__((cleanup(fr))) = 0; // obt-note {{jump bypasses initialization of variable with __attribute__((cleanup))}}
l:
  ;
}

// A handler only for the default label's checks can still be bypassed by the
// builtin's.
void builtin_only_checked(int __ob_trap a, int n) {
  int v[n];
  a = __builtin_overflow_goto(a + 1, l);
  v[0] = a;
l:
  ;
}

// Checks in a variable length array size trap, and the builtin may not branch
// out of one.
int builtin_in_vla_size(int __ob_trap n) {
  int v[__builtin_overflow_goto(n + 1, l)]; // obt-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                            // obt-note {{jump leaves a variable length array size}}
  v[0] = 1;
  return v[0];
l:
  return -1;
}
