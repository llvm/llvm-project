// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fsyntax-only -verify %s

void name_a(void) __attribute__((import_name)); //expected-error {{'import_name' attribute takes one argument}}

extern int name_b __attribute__((import_name("foo")));
int name_b_def __attribute__((import_name("foo"))); //expected-warning {{import name cannot be applied to a variable with a definition}}

void name_c(void) __attribute__((import_name("foo", "bar"))); //expected-error {{'import_name' attribute takes one argument}}

void name_d(void) __attribute__((import_name("foo", "bar", "qux"))); //expected-error {{'import_name' attribute takes one argument}}

void name_z(void) __attribute__((import_name("foo"))); //expected-note {{previous attribute is here}}

void name_z(void) __attribute__((import_name("bar"))); //expected-warning {{import name (bar) does not match the import name (foo) of the previous declaration}}

void module_a(void) __attribute__((import_module)); //expected-error {{'import_module' attribute takes one argument}}

extern int module_b __attribute__((import_module("foo")));
int module_b_def __attribute__((import_module("foo"))); //expected-warning {{import module cannot be applied to a variable with a definition}}

void module_c(void) __attribute__((import_module("foo", "bar"))); //expected-error {{'import_module' attribute takes one argument}}

void module_d(void) __attribute__((import_module("foo", "bar", "qux"))); //expected-error {{'import_module' attribute takes one argument}}

void module_z(void) __attribute__((import_module("foo"))); //expected-note {{previous attribute is here}}

void module_z(void) __attribute__((import_module("bar"))); //expected-warning {{import module (bar) does not match the import module (foo) of the previous declaration}}

void both(void) __attribute__((import_name("foo"), import_module("bar")));

// export_name tests
void export_a(void) __attribute__((export_name)); //expected-error {{'export_name' attribute takes one argument}}
extern int export_a_var __attribute__((export_name)); //expected-error {{'export_name' attribute takes one argument}}
void export_b(void) __attribute__((export_name("foo", "bar"))); //expected-error {{'export_name' attribute takes one argument}}

void export_c(void) __attribute__((export_name("foo"))); //expected-note {{previous attribute is here}}
void export_c(void) __attribute__((export_name("bar"))); //expected-warning {{export name (bar) does not match the export name (foo) of the previous declaration}}

extern int export_d __attribute__((export_name("foo"))); //expected-note {{previous attribute is here}}
extern int export_d __attribute__((export_name("bar"))); //expected-warning {{export name (bar) does not match the export name (foo) of the previous declaration}}

// Variable mismatch tests for import_module/name
extern int name_z_var __attribute__((import_name("foo"))); //expected-note {{previous attribute is here}}
extern int name_z_var __attribute__((import_name("bar"))); //expected-warning {{import name (bar) does not match the import name (foo) of the previous declaration}}

extern int module_z_var __attribute__((import_module("foo"))); //expected-note {{previous attribute is here}}
extern int module_z_var __attribute__((import_module("bar"))); //expected-warning {{import module (bar) does not match the import module (foo) of the previous declaration}}

// Explicit 'used' on non-definition still warns, while 'export_name' (implicit 'used') does not
extern int explicit_used_var __attribute__((used)); //expected-warning {{'used' attribute ignored on a non-definition declaration}}

struct WasmGlobalRecord {
	int value;
};

extern struct WasmGlobalRecord wasm_global_record
		__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute requires a scalar type}}
extern int wasm_global_array[2]
		__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute requires a scalar type}}
extern int wasm_global_value __attribute__((wasm_global));
extern int wasm_global_value;
int wasm_global_value = 0;
extern int *wasm_global_pointer __attribute__((wasm_global));
extern long long wasm_global_integer __attribute__((wasm_global));
extern unsigned _BitInt(64) wasm_global_bitint __attribute__((wasm_global));
extern double wasm_global_double __attribute__((wasm_global));
extern _Complex double wasm_global_complex
	__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute does not support type}}
extern long double wasm_global_long_double
	__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute does not support type}}
extern unsigned _BitInt(65) wasm_global_wide_bitint
	__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute does not support type}}
extern int __attribute__((address_space(1))) wasm_global_address_space
	__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute cannot be used with an explicit address space}}

_Thread_local int wasm_global_c_tls
		__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute cannot be applied to a thread-local variable}}
__thread int wasm_global_gnu_tls
		__attribute__((wasm_global)); // expected-error {{'wasm_global' attribute cannot be applied to a thread-local variable}}

void wasm_global_address_test(void) {
	int *pointer = &wasm_global_value; // expected-error {{cannot take the address of a WebAssembly global}}
}

