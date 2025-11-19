// Test 1: Without C11 and without flag - should NOT warn
// RUN: %clang_analyze_cc1 %s -verify -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -DEXPECT_NO_WARNINGS

// Test 2: Without C11 but with flag enabled - should warn
// RUN: %clang_analyze_cc1 %s -verify -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:AllowWithoutC11=true \
// RUN:   -DEXPECT_WARNINGS

// Test 3: With C11 - should warn (existing behavior)
// RUN: %clang_analyze_cc1 %s -verify -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -DEXPECT_WARNINGS

#include "Inputs/system-header-simulator.h"

extern char buf[128];
extern char src[128];

void test_memcpy(void) {
  memcpy(buf, src, 10);
#ifdef EXPECT_WARNINGS
  // expected-warning@-2{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard}}
#else
  // expected-no-diagnostics
#endif
}

void test_memset(void) {
  memset(buf, 0, 10);
#ifdef EXPECT_WARNINGS
  // expected-warning@-2{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard}}
#else
  // expected-no-diagnostics
#endif
}

void test_memmove(void) {
  memmove(buf, src, 10);
#ifdef EXPECT_WARNINGS
  // expected-warning@-2{{Call to function 'memmove' is insecure as it does not provide security checks introduced in the C11 standard}}
#else
  // expected-no-diagnostics
#endif
}

