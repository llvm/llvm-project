// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -DTEST -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fsyntax-only -verify %s

#if TEST
void (__attribute__((regparm(3), stdcall)) *pf) (); //expected-warning {{calling convention 'stdcall' ignored for this target}}
void (__attribute__((regparm(2), stdcall)) foo)(int a) { //expected-warning {{calling convention 'stdcall' ignored for this target}}
}
#else
//expected-no-diagnostics
void (__attribute__((regparm(3), stdcall)) *pf) ();
void (__attribute__((regparm(2), stdcall)) foo)(int a) {}
#endif
