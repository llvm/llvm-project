// RUN: %check_clang_tidy %s misc-unused-parameters %t -- -- -Wno-strict-prototypes

// Basic removal
// =============
void a(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void a(int i) {;}{{$}}

#if __STDC_VERSION__ < 202311L
static void b(); // In C before C23, forward declarations can leave out parameters.
#endif
static void b(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}static void b() {;}{{$}}

// Unchanged cases
// ===============
#if __STDC_VERSION__ < 202311L
void h(i, c, d) int i; char *c, *d; {} // Don't mess with K&R style
#endif

// Do not warn on naked functions.
__attribute__((naked)) void nakedFunction(int a, int b) { ; }
