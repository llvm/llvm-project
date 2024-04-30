// RUN: %clang_cc1 -fsyntax-only -std=c23 %s -verify

// Ensure that thread_local and _Thread_local are synonyms in C23 and both
// restrict local variables to be explicitly static or extern.
void func(void) {
  // FIXME: it would be nice if the diagnostic said 'thread_local' in this case.
  thread_local int i = 12;  // expected-error {{'_Thread_local' variables must have global storage}}
  _Thread_local int j = 13; // expected-error {{'_Thread_local' variables must have global storage}}

  static thread_local int k = 14;
  static _Thread_local int l = 15;

  extern thread_local int m;
  extern thread_local int n;
}

// This would previously fail because the tls models were different.
extern thread_local unsigned a;
_Thread_local unsigned a = 0;
