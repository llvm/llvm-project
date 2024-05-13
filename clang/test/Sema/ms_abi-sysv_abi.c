// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-pc-linux-gnu -Wno-strict-prototypes %s

// CC qualifier can be applied only to functions
int __attribute__((ms_abi)) var1; // expected-warning{{'ms_abi' only applies to function types; type here is 'int'}}
int __attribute__((sysv_abi)) var2; // expected-warning{{'sysv_abi' only applies to function types; type here is 'int'}}

// Different CC qualifiers are not compatible
// FIXME: Should say 'sysv_abi' instead of 'cdecl'
void __attribute__((ms_abi, sysv_abi)) foo3(void); // expected-error{{cdecl and ms_abi attributes are not compatible}}
void __attribute__((ms_abi)) foo4(); // expected-note{{previous declaration is here}}
void __attribute__((sysv_abi)) foo4(void); // expected-error{{function declared 'cdecl' here was previously declared 'ms_abi'}}

void bar(int i, int j) __attribute__((ms_abi, cdecl)); // expected-error{{cdecl and ms_abi attributes are not compatible}}
void bar2(int i, int j) __attribute__((sysv_abi, cdecl)); // no-error
