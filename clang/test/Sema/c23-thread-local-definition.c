// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c17 -fsyntax-only -verify=pre23 %s
// pre23-no-diagnostics

_Thread_local int a;     // expected-note {{previous definition is here}}
_Thread_local int a;     // expected-error {{redefinition of 'a'}}

static _Thread_local int b; // expected-note {{previous definition is here}}
static _Thread_local int b; // expected-error {{redefinition of 'b'}}

__thread int c;          // expected-note {{previous definition is here}}
__thread int c;          // expected-error {{redefinition of 'c'}}

// 'extern' makes these declarations, not definitions.
extern _Thread_local int e;
_Thread_local int e;

_Thread_local int f;
extern _Thread_local int f;

#if __STDC_VERSION__ >= 202311L
thread_local int x, x;   // expected-error {{redefinition of 'x'}} \
                         // expected-note {{previous definition is here}}
#endif
