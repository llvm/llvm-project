// RUN: %clang_analyze_cc1 -verify=untrusted-env %s \
// RUN:   -analyzer-checker=core                    \
// RUN:   -analyzer-checker=optin.taint    \
// RUN:   -analyzer-checker=debug.TaintTest

// RUN: %clang_analyze_cc1 -verify %s -DEXPECT_NO_WARNINGS    \
// RUN:   -analyzer-config assume-controlled-environment=true \
// RUN:   -analyzer-checker=core                              \
// RUN:   -analyzer-checker=optin.taint              \
// RUN:   -analyzer-checker=debug.TaintTest


#ifdef EXPECT_NO_WARNINGS
// expected-no-diagnostics
#endif

char *getenv(const char *name);

void foo(void) {
  char *p = getenv("FOO"); // untrusted-env-warning {{tainted}}
  (void)p;                 // untrusted-env-warning {{tainted}}
}
