// RUN: %clang_cc1 %s -fsyntax-only -triple i386-unknown-unknown -verify

void __attribute__((fastcall)) foo(float *a) { 
}

void __attribute__((stdcall)) bar(float *a) { 
}

void __attribute__((fastcall(1))) baz(float *a) { // expected-error {{attribute takes no arguments}}
}

void __attribute__((fastcall)) test0() { // expected-error {{function with no prototype cannot use fastcall calling convention}}
}

void __attribute__((fastcall)) test1(void) {
}

void __attribute__((fastcall)) test2(int a, ...) { // expected-error {{variadic function cannot use fastcall calling convention}}
}

void __attribute__((cdecl)) ctest0() {}

void __attribute__((cdecl(1))) ctest1(float x) {} // expected-error {{attribute takes no arguments}}

void (__attribute__((fastcall)) *pfoo)(float*) = foo;

void (__attribute__((stdcall)) *pbar)(float*) = bar;

void (__attribute__((cdecl)) *ptest1)(void) = test1; // expected-warning {{incompatible pointer types}}

void (*pctest0)() = ctest0;

void ctest2() {}
void (__attribute__((cdecl)) *pctest2)() = ctest2;

typedef void (__attribute__((fastcall)) *Handler) (float *);
Handler H = foo;

int __attribute__((pcs("aapcs", "aapcs"))) pcs1(void); // expected-error {{attribute takes one argument}}
int __attribute__((pcs())) pcs2(void); // expected-error {{attribute takes one argument}}
int __attribute__((pcs(pcs1))) pcs3(void); // expected-error {{attribute takes one argument}}
int __attribute__((pcs(0))) pcs4(void); // expected-error {{'pcs' attribute requires parameter 1 to be a string}}
/* These are ignored because the target is i386 and not ARM */
int __attribute__((pcs("aapcs"))) pcs5(void); // expected-warning {{calling convention 'pcs' ignored for this target}}
int __attribute__((pcs("aapcs-vfp"))) pcs6(void); // expected-warning {{calling convention 'pcs' ignored for this target}}
int __attribute__((pcs("foo"))) pcs7(void); // expected-error {{Invalid PCS type}}

// PR6361
void ctest3();
void __attribute__((cdecl)) ctest3() {}

// PR6408
typedef __attribute__((stdcall)) void (*PROC)();
PROC __attribute__((cdecl)) ctest4(const char *x) {}

void __attribute__((pnaclcall)) pnaclfunc(float *a) {} // expected-warning {{calling convention 'pnaclcall' ignored for this target}}
