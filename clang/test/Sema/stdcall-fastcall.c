// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify -triple i686-apple-darwin10 %s

// CC qualifier can be applied only to functions
int __attribute__((stdcall)) var1; // expected-warning{{'stdcall' only applies to function types; type here is 'int'}}
int __attribute__((fastcall)) var2; // expected-warning{{'fastcall' only applies to function types; type here is 'int'}}

// Different CC qualifiers are not compatible
void __attribute__((stdcall, fastcall)) foo3(void); // expected-error{{fastcall and stdcall attributes are not compatible}}
void __attribute__((stdcall)) foo4(); // expected-note{{previous declaration is here}} expected-warning{{function with no prototype cannot use the stdcall calling convention}}
void __attribute__((fastcall)) foo4(void); // expected-error{{function declared 'fastcall' here was previously declared 'stdcall'}}

// rdar://8876096
void rdar8876096foo1(int i, int j) __attribute__((fastcall, cdecl)); // expected-error {{not compatible}}
void rdar8876096foo2(int i, int j) __attribute__((fastcall, stdcall)); // expected-error {{not compatible}}
void rdar8876096foo3(int i, int j) __attribute__((fastcall, regparm(2))); // expected-error {{not compatible}}
void rdar8876096foo4(int i, int j) __attribute__((stdcall, cdecl)); // expected-error {{not compatible}}
void rdar8876096foo5(int i, int j) __attribute__((stdcall, fastcall)); // expected-error {{not compatible}}
void rdar8876096foo6(int i, int j) __attribute__((cdecl, fastcall)); // expected-error {{not compatible}}
void rdar8876096foo7(int i, int j) __attribute__((cdecl, stdcall)); // expected-error {{not compatible}}
void rdar8876096foo8(int i, int j) __attribute__((regparm(2), fastcall)); // expected-error {{not compatible}}
