// RUN: %clang_cc1 %s -fsyntax-only -Wno-strict-prototypes -triple i386-unknown-unknown -verify
// RUN: %clang_cc1 %s -fsyntax-only -Wno-strict-prototypes -triple i386-unknown-unknown -fms-compatibility -DWIN -verify

void __attribute__((fastcall)) foo(float *a) {
}

void __attribute__((stdcall)) bar(float *a) {
}

void __attribute__((fastcall(1))) baz(float *a) { // expected-error {{'fastcall' attribute takes no arguments}}
}

void __attribute__((fastcall)) test0() {
}

void __attribute__((fastcall)) test1(void) {
}

void __attribute__((fastcall)) test2(int a, ...) { // expected-warning {{fastcall calling convention is not supported on variadic function}}
}
void __attribute__((stdcall)) test3(int a, ...) { // expected-warning {{stdcall calling convention is not supported on variadic function}}
}
void __attribute__((thiscall)) test4(int a, ...) { // expected-error {{variadic function cannot use thiscall calling convention}}
}

void __attribute__((cdecl)) ctest0() {}

void __attribute__((cdecl(1))) ctest1(float x) {} // expected-error {{'cdecl' attribute takes no arguments}}

void (__attribute__((fastcall)) *pfoo)(float*) = foo;

void (__attribute__((stdcall)) *pbar)(float*) = bar;

void (__attribute__((cdecl)) *ptest1)(void) = test1; // expected-warning {{incompatible function pointer types}}

void (*pctest0)() = ctest0;

void ctest2() {}
void (__attribute__((cdecl)) *pctest2)() = ctest2;

typedef void (__attribute__((fastcall)) *Handler) (float *);
Handler H = foo;

int __attribute__((pcs("aapcs", "aapcs"))) pcs1(void); // expected-error {{'pcs' attribute takes one argument}}
int __attribute__((pcs())) pcs2(void); // expected-error {{'pcs' attribute takes one argument}}
int __attribute__((pcs(pcs1))) pcs3(void); // expected-error {{'pcs' attribute requires a string}} \
                                           // expected-error {{invalid PCS type}}
int __attribute__((pcs(0))) pcs4(void); // expected-error {{'pcs' attribute requires a string}}
/* These are ignored because the target is i386 and not ARM */
int __attribute__((pcs("aapcs"))) pcs5(void); // expected-warning {{'pcs' calling convention is not supported for this target}}
int __attribute__((pcs("aapcs-vfp"))) pcs6(void); // expected-warning {{'pcs' calling convention is not supported for this target}}
int __attribute__((pcs("foo"))) pcs7(void); // expected-error {{invalid PCS type}}

int __attribute__((aarch64_vector_pcs)) aavpcs(void); // expected-warning {{'aarch64_vector_pcs' calling convention is not supported for this target}}

// PR6361
void ctest3();
void __attribute__((cdecl)) ctest3() {}

// PR6408
typedef __attribute__((stdcall)) void (*PROC)();
PROC __attribute__((cdecl)) ctest4(const char *x) {}

void __attribute__((intel_ocl_bicc)) inteloclbifunc(float *a) {}

typedef void typedef_fun_t(int);
typedef_fun_t typedef_fun; // expected-note {{previous declaration is here}}
void __attribute__((stdcall)) typedef_fun(int x) { } // expected-error {{function declared 'stdcall' here was previously declared without calling convention}}

struct type_test {} __attribute__((stdcall));  // expected-warning {{'stdcall' attribute only applies to functions and methods}}

void __vectorcall __builtin_unreachable(); // expected-warning {{vectorcall calling convention is not supported on builtin function}}
