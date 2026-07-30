// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fsyntax-only -verify %s

void name_a(void) __attribute__((import_name)); //expected-error {{'import_name' attribute takes one argument}}

int name_b __attribute__((import_name("foo"))); //expected-error {{'import_name' attribute only applies to functions}}

void name_c(void) __attribute__((import_name("foo", "bar"))); //expected-error {{'import_name' attribute takes one argument}}

void name_d(void) __attribute__((import_name("foo", "bar", "qux"))); //expected-error {{'import_name' attribute takes one argument}}

void name_z(void) __attribute__((import_name("foo"))); //expected-note {{previous attribute is here}}

void name_z(void) __attribute__((import_name("bar"))); //expected-warning {{import name (bar) does not match the import name (foo) of the previous declaration}}

void module_a(void) __attribute__((import_module)); //expected-error {{'import_module' attribute takes one argument}}

int module_b __attribute__((import_module("foo"))); //expected-error {{'import_module' attribute only applies to functions}}

void module_c(void) __attribute__((import_module("foo", "bar"))); //expected-error {{'import_module' attribute takes one argument}}

void module_d(void) __attribute__((import_module("foo", "bar", "qux"))); //expected-error {{'import_module' attribute takes one argument}}

void module_z(void) __attribute__((import_module("foo"))); //expected-note {{previous attribute is here}}

void module_z(void) __attribute__((import_module("bar"))); //expected-warning {{import module (bar) does not match the import module (foo) of the previous declaration}}

void both(void) __attribute__((import_name("foo"), import_module("bar")));
