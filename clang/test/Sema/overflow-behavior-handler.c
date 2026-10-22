// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -Wunused-label -verify %s

void fr(int *);

// A handler label that a checked operation can reach is used.
int used(int __ob_trap a) {
  a = a + 1;
  return a;
__overflow:
  return -1;
}

// A handler label after the checked operation in source order is used too.
int used_before(int __ob_trap a) {
  if (a)
    goto __overflow;
  return a * 2;
__overflow:
  return -1;
}

// No checked operation: the label is unused.
int unused_label(int a) {
  return a + 1;
__overflow: // expected-warning {{unused label '__overflow'}}
  return -1;
}

// Wrapping arithmetic is never checked.
int wrap(int __ob_wrap a) {
  return a + 1;
__overflow: // expected-warning {{unused label '__overflow'}}
  return -1;
}

// Only the configured name (default __overflow) is a handler.
int other_name(int __ob_trap a) {
  return a + 1;
efault: // expected-warning {{unused label 'efault'}}
  return -1;
}

// The handler jump may not enter the scope of a VLA or a cleanup variable.
int into_vla(int __ob_trap n) {
  n = n + 1; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int v[n];  // expected-note {{jump bypasses initialization of variable length array}}
  v[0] = 0;
  return v[0];
__overflow:
  return -1;
}

int into_cleanup(int __ob_trap n) {
  n++;       // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int z __attribute__((cleanup(fr))) = 0; // expected-note {{jump bypasses initialization of variable with __attribute__((cleanup))}}
  return n;
__overflow:
  return -1;
}

// Leaving a cleanup scope is fine.
int out_of_cleanup(int __ob_trap n) {
  {
    int z __attribute__((cleanup(fr))) = 0;
    n *= 2;
  }
  return n;
__overflow:
  return -1;
}

// Without a handler there is no jump, so nothing to diagnose.
int no_handler(int __ob_trap n) {
  n = n + 1;
  int v[n];
  v[0] = 0;
  return v[0];
}

// A GNU local label redirects checks inside its block only; the operation in
// the statement expression does not jump past the VLA.
int local_label(int __ob_trap n) {
  int r = ({
    __label__ __overflow;
    int t = n * 2;
    goto done;
  __overflow:
    t = -2;
  done:
    t;
  });
  int v[r + 1];
  v[0] = r;
  return v[0];
}

// Checked conversions jump too.
int conv_into_vla(unsigned long long __ob_trap v, int n) {
  unsigned char c = v; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int vla[n];          // expected-note {{jump bypasses initialization of variable length array}}
  vla[0] = c;
  return vla[0];
__overflow:
  return -1;
}

int sign_into_vla(int __ob_trap v, int n) {
  unsigned u = v; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int vla[n];     // expected-note {{jump bypasses initialization of variable length array}}
  vla[0] = u;
  return vla[0];
__overflow:
  return -1;
}

int shift_into_vla(unsigned char __ob_trap x, int n) {
  x <<= 4; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  int vla[n]; // expected-note {{jump bypasses initialization of variable length array}}
  vla[0] = x;
  return vla[0];
__overflow:
  return -1;
}

// Widening conversions are never checked.
long widen(int __ob_trap v, int n) {
  long l = v;
  int vla[n];
  vla[0] = 0;
  return l + vla[0];
__overflow: // expected-warning {{unused label '__overflow'}}
  return -1;
}
