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
