// RUN: %check_clang_tidy %s readability-redundant-preprocessor %t -- -- -I %S

// Positive testing.
#ifndef FOO
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #ifndef; consider removing it [readability-redundant-preprocessor]
#ifndef FOO
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifndef was here
void f();
#endif
#endif

// Positive testing of inverted condition.
#ifndef FOO
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #ifdef; consider removing it [readability-redundant-preprocessor]
#ifdef FOO
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifndef was here
void f2();
#endif
#endif

// Negative testing.
#include "redundant-preprocessor.h"

#ifndef BAR
void g();
#endif

#ifndef FOO
#ifndef BAR
void h();
#endif
#endif

#ifndef FOO
#ifdef BAR
void i();
#endif
#endif

// Positive #if testing.
#define FOO 4

#if FOO == 4
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#if FOO == 4
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
void j();
#endif
#endif

#if FOO == 3 + 1
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#if FOO == 3 + 1
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
void j();
#endif
#endif

#if FOO == \
    4
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#if FOO == \
    4
// CHECK-NOTES: [[@LINE-5]]:2: note: previous #if was here
void j();
#endif
#endif

// Negative #if testing.
#define BAR 4

#if FOO == 4
#if BAR == 4
void k();
#endif
#endif

#if FOO == \
    4
#if BAR == \
    5
void k();
#endif
#endif

// Different builtin checks should NOT trigger warning
#if __has_builtin(__remove_cvref)
#  if __has_cpp_attribute(no_unique_address)
#  endif
#endif

// Redundant nested #if
#if defined(FOO)
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#  if defined(FOO)
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
#  endif
#endif

// Different __has_builtin checks
#if __has_builtin(__builtin_assume)
#  if __has_builtin(__builtin_expect)
#  endif
#endif

// Different __has_cpp_attribute checks
#if __has_cpp_attribute(fallthrough)
#  if __has_cpp_attribute(nodiscard)
#  endif
#endif

// Same __has_builtin check
#if __has_builtin(__remove_cvref)
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#  if __has_builtin(__remove_cvref)
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
#  endif
#endif

// Different __has_include checks
#if __has_include(<vector>)
#  if __has_include(<string>)
#  endif
#endif

// Complex expressions - different conditions
#if __has_builtin(__builtin_clz) && defined(__GNUC__)
#  if __has_builtin(__builtin_ctz) && defined(__GNUC__)
#  endif
#endif

#define MACRO1
#define MACRO2

// Redundant #ifdef
#ifdef MACRO1
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #ifdef; consider removing it [readability-redundant-preprocessor]
#  ifdef MACRO1
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifdef was here
#  endif
#endif

// Different macros - #ifdef
#ifdef MACRO1
#  ifdef MACRO2
#  endif
#endif

// Redundant #ifndef
#ifndef MACRO3
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #ifndef; consider removing it [readability-redundant-preprocessor]
#  ifndef MACRO3
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifndef was here
#  endif
#endif

// Different macros - #ifndef
#ifndef MACRO3
#  ifndef MACRO4
#  endif
#endif

// Different __has_feature checks
#if __has_feature(cxx_rvalue_references)
#  if __has_feature(cxx_lambdas)
#  endif
#endif

// Different version checks
#if __cplusplus >= 201103L
#  if __cplusplus >= 201402L
#  endif
#endif

// Different builtin functions with similar names
#if __has_builtin(__builtin_addressof)
#  if __has_builtin(__builtin_assume_aligned)
#  endif
#endif

// Different compiler checks
#if defined(__clang__)
#  if defined(__GNUC__)
#  endif
#endif

// Mixed builtin and regular macro
#if __has_builtin(__make_integer_seq)
#  if defined(USE_STD_INTEGER_SEQUENCE)
#  endif
#endif

// Different numeric comparisons
#if __GNUC__ >= 2
#  if __GNUC__ >= 3
#  endif
#endif

// Test: inline comments - same condition and comment SHOULD warn
#if __has_builtin(__remove_cvref) // same comment
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#  if __has_builtin(__remove_cvref) // same comment
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
#  endif
#endif

// Test: inline comments - different comments, condition text differs, no warn
#if defined(FOO) // comment one
#  if defined(FOO) // comment two
#  endif
#endif

// Test: block comments - same condition and comment SHOULD warn
#if __has_builtin(__remove_cvref) /* block */
// CHECK-NOTES: [[@LINE+1]]:4: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
#  if __has_builtin(__remove_cvref) /* block */
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
#  endif
#endif

// Test: multiline block comment spanning lines - same condition SHOULD warn
#if __has_builtin(__remove_cvref) /* multiline
                                     comment */
// CHECK-NOTES: [[@LINE+2]]:4: warning: nested redundant #if; consider removing it [readability-redundant-preprocessor]
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #if was here
#  if __has_builtin(__remove_cvref) /* multiline
                                     comment */
#  endif
#endif

// Test: multiline block comment - different conditions should NOT warn
#if __has_builtin(__remove_cvref) /* multiline
                                     comment */
#  if __has_cpp_attribute(no_unique_address) /* multiline
                                                comment */
#  endif
#endif
