// Verify that -mloadtime-comment-vars= diagnoses named C variables it cannot
// preserve. The diagnostics are produced by Sema and fire even with
// -fsyntax-only.
//
// Covered cases:
//   - volatile-qualified pointer (char *volatile)
//   - pointer-to-volatile character (volatile char *)
//   - volatile character array (volatile char[])
//   - thread-local variable (__thread, not static storage duration)
//   - pointer not initialized directly with a string literal
//   - valid const char array — no diagnostic

// RUN: %clang_cc1 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=vol_ptr,vol_char,vol_arr,tls_ptr,ind_ptr,const_arr,lfn \
// RUN:   -fsyntax-only -verify %s

// A volatile-qualified pointer is diagnosed.
char *volatile vol_ptr = "@(#) vol ptr"; // expected-warning {{'vol_ptr' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// A pointer to volatile characters is diagnosed.
volatile char *vol_char = "@(#) vol char"; // expected-warning {{'vol_char' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// A volatile character array is diagnosed.
volatile char vol_arr[] = "@(#) vol arr"; // expected-warning {{'vol_arr' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// A thread-local variable does not have static storage duration and is
// diagnosed.
__thread char *tls_ptr = "@(#) tls"; // expected-warning {{'tls_ptr' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

// A pointer bound to another object rather than a string literal is
// diagnosed.
static const char target[] = "@(#) target";
const char *ind_ptr = target; // expected-warning {{'ind_ptr' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// A const character array is a valid form; no diagnostic is expected.
const char const_arr[] = "@(#) const arr";

// A function-local static: name-matched, so diagnosed rather than silently
// ignored.
void h(void) { static char lfn[] = "@(#) lfn"; (void)lfn; } // expected-warning {{'lfn' named in '-mloadtime-comment-vars=' is a function-local variable and will not be preserved}}
