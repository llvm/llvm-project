// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-pc-linux-gnu %s

// CC qualifier can be applied only to functions
int __attribute__((stdcall)) var1; // expected-warning{{'stdcall' only applies to function types; type here is 'int'}}
int __attribute__((fastcall)) var2; // expected-warning{{'fastcall' only applies to function types; type here is 'int'}}

// Different CC qualifiers are not compatible
void __attribute__((stdcall, fastcall)) foo3(void); // expected-warning{{calling convention 'stdcall' ignored for this target}} expected-warning {{calling convention 'fastcall' ignored for this target}}
void __attribute__((stdcall)) foo4(); // expected-warning{{calling convention 'stdcall' ignored for this target}}
void __attribute__((fastcall)) foo4(void); // expected-warning {{calling convention 'fastcall' ignored for this target}}

// rdar://8876096
void rdar8876096foo1(int i, int j) __attribute__((fastcall, cdecl)); // expected-warning{{calling convention 'fastcall' ignored for this target}}
void rdar8876096foo2(int i, int j) __attribute__((fastcall, stdcall)); // expected-warning{{calling convention 'stdcall' ignored for this target}} expected-warning {{calling convention 'fastcall' ignored for this target}}
void rdar8876096foo3(int i, int j) __attribute__((fastcall, regparm(2))); // expected-warning {{calling convention 'fastcall' ignored for this target}}
void rdar8876096foo4(int i, int j) __attribute__((stdcall, cdecl)); // expected-warning{{calling convention 'stdcall' ignored for this target}}
void rdar8876096foo5(int i, int j) __attribute__((stdcall, fastcall)); // expected-warning{{calling convention 'stdcall' ignored for this target}} expected-warning {{calling convention 'fastcall' ignored for this target}}
void rdar8876096foo6(int i, int j) __attribute__((cdecl, fastcall)); // expected-warning {{calling convention 'fastcall' ignored for this target}}
void rdar8876096foo7(int i, int j) __attribute__((cdecl, stdcall)); // expected-warning{{calling convention 'stdcall' ignored for this target}}
void rdar8876096foo8(int i, int j) __attribute__((regparm(2), fastcall)); // expected-warning {{calling convention 'fastcall' ignored for this target}}
