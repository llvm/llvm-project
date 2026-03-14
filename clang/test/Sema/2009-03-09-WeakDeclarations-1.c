// RUN: %clang_cc1 -verify %s -triple i686-apple-darwin
// Insist upon warnings for inappropriate weak attributes.

// O.K.
extern int ext_weak_import __attribute__ ((__weak_import__));

// These are inappropriate, and should generate warnings:
int decl_weak_import __attribute__ ((__weak_import__)); // expected-warning {{'weak_import' attribute cannot be specified on a definition}}
int decl_initialized_weak_import __attribute__ ((__weak_import__)) = 13; // expected-warning {{'weak_import' attribute cannot be specified on a definition}}

// O.K.
extern int ext_f(void) __attribute__ ((__weak_import__));

// These are inappropriate, and should generate warnings:
int def_f(void) __attribute__ ((__weak_import__));
int __attribute__ ((__weak_import__)) decl_f(void) {return 0;};
