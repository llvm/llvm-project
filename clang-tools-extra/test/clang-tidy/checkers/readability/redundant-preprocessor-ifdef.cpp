// RUN: %check_clang_tidy %s readability-redundant-preprocessor %t -- -- -DFOO

// Positive testing.
#ifdef FOO
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #ifdef; consider removing it [readability-redundant-preprocessor]
#ifdef FOO
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifdef was here
void f();
#endif
#endif

// Positive testing of inverted condition.
#ifdef FOO
// CHECK-NOTES: [[@LINE+1]]:2: warning: nested redundant #ifndef; consider removing it [readability-redundant-preprocessor]
#ifndef FOO
// CHECK-NOTES: [[@LINE-3]]:2: note: previous #ifdef was here
void f2();
#endif
#endif

// Negative testing.
#ifdef BAR
void g();
#endif

#ifdef FOO
#ifdef BAR
void h();
#endif
#endif

#ifdef FOO
#ifndef BAR
void i();
#endif
#endif
