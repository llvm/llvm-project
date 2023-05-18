// RUN: %clang_cc1 -fsyntax-only %s -verify
#ifndef __has_warning
#error Should have __has_warning
#endif

#if __has_warning("not valid") // expected-warning {{__has_warning expected option name}}
#endif

// expected-warning@+2 {{Should have -Wparentheses}}
#if __has_warning("-Wparentheses")
#warning Should have -Wparentheses
#endif

// expected-error@+2 {{expected string literal in '__has_warning'}}
// expected-error@+1 {{missing ')'}} expected-note@+1 {{match}}
#if __has_warning(-Wfoo)
#endif

// expected-warning@+3 {{Not a valid warning flag}}
#if __has_warning("-Wnot-a-valid-warning-flag-at-all")
#else
#warning Not a valid warning flag
#endif

// expected-error@+1 {{missing '(' after '__has_warning'}}
#if __has_warning "not valid"
#endif

// Macro expansion does not occur in the parameter to __has_warning
// (as is also expected behaviour for ordinary macros), so the
// following should not expand:

#define MY_ALIAS "-Wparentheses"

// expected-error@+1 {{expected}}
#if __has_warning(MY_ALIAS)
#error Alias expansion not allowed
#endif

// But deferring should expand:
#define HAS_WARNING(X) __has_warning(X)

#if !HAS_WARNING(MY_ALIAS)
#error Expansion should have occurred
#endif
