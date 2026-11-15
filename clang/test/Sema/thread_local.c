// RUN: %clang_cc1 -fsyntax-only -std=c23 %s -verify
// UNSUPPORTED: target={{.*}}-zos{{.*}}

// Ensure that thread_local and _Thread_local are synonyms in C23 and both
// restrict local variables to be explicitly static or extern.
void func(void) {
  thread_local int i = 12; // expected-error {{'thread_local' variables must have global storage}}
  _Thread_local int j = 13; // expected-error {{'_Thread_local' variables must have global storage}}

  static thread_local int k = 14;
  static _Thread_local int l = 15;

  extern thread_local int m;
  extern thread_local int n;
}

// This would previously fail because the tls models were different.
extern thread_local unsigned a;
_Thread_local unsigned a = 0;

thread_local _Thread_local int duplicate1;
// expected-warning@-1 {{duplicate 'thread_local' declaration specifier}}
_Thread_local thread_local int duplicate2;
// expected-warning@-1 {{duplicate '_Thread_local' declaration specifier}}
