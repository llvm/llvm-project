// Verify that -mloadtime-comment-vars= diagnoses named C variables it cannot
// preserve. The diagnostics are produced by Sema and fire even with
// -fsyntax-only.
//
// Covered cases:
//   - volatile-qualified pointer (char *volatile)
//   - pointer-to-volatile character (volatile char *)
//   - volatile character array (volatile char[])
//   - thread-local variable (__thread, not static storage duration)
//   - pointer not initialized directly with a string literal, including
//     when the initializer is enclosed in braces; a braced string literal is
//     accepted without diagnostic
//   - name-matched variables of an unsupported type (int, unsigned char)
//   - function-local static (name-matched, so diagnosed rather than ignored)
//   - variables defined as an alias (array and pointer) are diagnosed, while
//     naming the aliased variable is accepted without diagnostic
//   - valid const char array — no diagnostic
//   - a preserved variable counts as used: no -Wunused-const-variable for it,
//     while an unlisted static const variable still warns
//
// The NONAIX scenario invokes cc1 directly for a non-AIX target: the option
// is rejected with an error.

// RUN: %clang_cc1 -triple powerpc64-ibm-aix -Wunused-const-variable \
// RUN:   -mloadtime-comment-vars=vol_ptr,vol_char,vol_arr,tls_ptr,ind_ptr,const_arr,lfn,kept,notchar,ustr,alias_arr,alias_ptr,aliasee_arr,braced_ok,braced_ind \
// RUN:   -fsyntax-only -verify %s

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -mloadtime-comment-vars=sccsid \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=NONAIX %s
// NONAIX: error: unsupported option '-mloadtime-comment-vars=' for target 'x86_64-unknown-linux-gnu'

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

// Braces around the initializer do not change the outcome: a braced string
// literal is accepted without diagnostic, and a braced pointer to another
// object is diagnosed.
const char *braced_ok = {"@(#) braced"};
const char *braced_ind = {target}; // expected-warning {{'braced_ind' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// A const character array is a valid form; no diagnostic is expected.
const char const_arr[] = "@(#) const arr";

// A function-local static: name-matched, so diagnosed rather than silently
// ignored.
void h(void) { static char lfn[] = "@(#) lfn"; (void)lfn; } // expected-warning {{'lfn' named in '-mloadtime-comment-vars=' is a function-local variable and will not be preserved}}

// A preserved variable is materially used — it is forced into the object
// file — so -Wunused-const-variable does not fire for it.
static const char kept[] = "@(#) kept";

// An unlisted static const variable is unaffected by the option and still
// gets the unused warning.
static const char dropped[] = "@(#) dropped"; // expected-warning {{unused variable 'dropped'}}

// A name match on a variable of any other type still demonstrates intent, so
// it is diagnosed rather than silently ignored.
static int notchar = 42; // expected-warning {{'notchar' named in '-mloadtime-comment-vars=' does not have a plain char pointer or array type and will not be preserved}}
static unsigned char ustr[] = "@(#) u"; // expected-warning {{'ustr' named in '-mloadtime-comment-vars=' does not have a plain char pointer or array type and will not be preserved}}

// A variable defined as an alias has no storage of its own and is diagnosed.
// The aliased variable is the one to name: aliasee_arr is listed and accepted
// without diagnostic.
char aliasee_arr[] = "@(#) aliasee arr";
extern char alias_arr[17] __attribute__((alias("aliasee_arr"))); // expected-warning {{'alias_arr' named in '-mloadtime-comment-vars=' is defined as an alias and will not be preserved}}
const char *aliasee_ptr = "@(#) aliasee ptr";
extern const char *alias_ptr __attribute__((alias("aliasee_ptr"))); // expected-warning {{'alias_ptr' named in '-mloadtime-comment-vars=' is defined as an alias and will not be preserved}}
