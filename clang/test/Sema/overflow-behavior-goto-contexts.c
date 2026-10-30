// RUN: %clang_cc1 -fsyntax-only -triple x86_64-windows-msvc -fms-extensions -std=c23 -fdefer-ts -Wno-redundant-defer -fexperimental-overflow-behavior-types -verify %s

// The builtin may not branch out of code that cannot be left.

int finally_block(int __ob_trap a) {
  __try {
  } __finally {
    a = __builtin_overflow_goto(a + 1, l); // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}}
  } // expected-note@-2 {{jump leaves a __finally block}}
  return a;
l:
  return -1;
}

int except_filter(int __ob_trap a) {
  __try {
  } __except (__builtin_overflow_goto(a + 1, l)) { // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                                   // expected-note {{jump leaves an __except filter}}
  }
  return a;
l:
  return -1;
}

int defer_body(int __ob_trap a) {
  {
    _Defer { a = __builtin_overflow_goto(a + 1, l); } // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                                      // expected-note {{jump exits a defer statement}}
  }
  return a;
l:
  return -1;
}
