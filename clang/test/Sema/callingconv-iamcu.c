// RUN: %clang_cc1 %s -fsyntax-only -triple i686-intel-elfiamcu -verify

void __attribute__((fastcall)) foo(float *a) { // expected-warning {{calling convention 'fastcall' ignored for this target}}
}

void __attribute__((stdcall)) bar(float *a) { // expected-warning {{calling convention 'stdcall' ignored for this target}}
}

void __attribute__((fastcall(1))) baz(float *a) { // expected-error {{'fastcall' attribute takes no arguments}}
}

void __attribute__((fastcall)) test2(int a, ...) { // expected-warning {{calling convention 'fastcall' ignored for this target}}
}
void __attribute__((stdcall)) test3(int a, ...) { // expected-warning {{calling convention 'stdcall' ignored for this target}}
}
void __attribute__((thiscall)) test4(int a, ...) { // expected-warning {{calling convention 'thiscall' ignored for this target}}
}

void __attribute__((cdecl)) ctest0() {}

void __attribute__((cdecl(1))) ctest1(float x) {} // expected-error {{'cdecl' attribute takes no arguments}}

void (__attribute__((fastcall)) *pfoo)(float*) = foo; // expected-warning {{calling convention 'fastcall' ignored for this target}}

void (__attribute__((stdcall)) *pbar)(float*) = bar; // expected-warning {{calling convention 'stdcall' ignored for this target}}

void (*pctest0)() = ctest0;

void ctest2() {}
void (__attribute__((cdecl)) *pctest2)() = ctest2;

typedef void (__attribute__((fastcall)) *Handler) (float *); // expected-warning {{calling convention 'fastcall' ignored for this target}}
Handler H = foo;

int __attribute__((pcs("aapcs", "aapcs"))) pcs1(void); // expected-error {{'pcs' attribute takes one argument}}
int __attribute__((pcs())) pcs2(void); // expected-error {{'pcs' attribute takes one argument}}
int __attribute__((pcs(pcs1))) pcs3(void); // expected-error {{'pcs' attribute requires a string}} \
                                           // expected-error {{invalid PCS type}}
int __attribute__((pcs(0))) pcs4(void); // expected-error {{'pcs' attribute requires a string}}
/* These are ignored because the target is i386 and not ARM */
int __attribute__((pcs("aapcs"))) pcs5(void); // expected-warning {{calling convention 'pcs' ignored for this target}}
int __attribute__((pcs("aapcs-vfp"))) pcs6(void); // expected-warning {{calling convention 'pcs' ignored for this target}}
int __attribute__((pcs("foo"))) pcs7(void); // expected-error {{invalid PCS type}}

void ctest3();
void __attribute__((cdecl)) ctest3() {}

typedef __attribute__((stdcall)) void (*PROC)(); // expected-warning {{calling convention 'stdcall' ignored for this target}}
PROC __attribute__((cdecl)) ctest4(const char *x) {}

void __attribute__((intel_ocl_bicc)) inteloclbifunc(float *a) {} // expected-warning {{calling convention 'intel_ocl_bicc' ignored for this target}}

struct type_test {} __attribute__((stdcall)); // expected-warning {{calling convention 'stdcall' ignored for this target}} expected-warning {{'stdcall' attribute only applies to functions and methods}}
