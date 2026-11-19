// RUN: %clang_cc1 -triple=x86_64-unknown-linux -Wpragmas -verify %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix-xcoff -Wpragmas -verify %s

// Check that pragma redefine_extname applies to external declarations only.
#pragma redefine_extname foo_static bar_static
static int foo_static(void) { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_static'}}

// Computing whether the declarations have external C linkage must not leave a
// stale linkage cached before they are connected to the preceding static
// declarations.
#pragma redefine_extname variable_after_static variable_alias
static int variable_after_static; // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to variable 'variable_after_static'}}
extern int variable_after_static;

#pragma redefine_extname function_after_static function_alias
static int function_after_static(void); // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'function_after_static'}}
extern int function_after_static(void);

unsigned __int128_t; // expected-error {{redefinition of '__int128_t' as different kind of symbol}}
