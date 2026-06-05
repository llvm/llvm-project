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
