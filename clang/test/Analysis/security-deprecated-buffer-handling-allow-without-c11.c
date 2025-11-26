// Test 1: Without C11 and without flag - should NOT warn
// RUN: %clang_analyze_cc1 %s -verify=c99-noflag -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

// Test 2: Without C11 but with flag enabled - should warn
// RUN: %clang_analyze_cc1 %s -verify=c99-withflag -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:AllowWithoutC11=true

// Test 3: With C11 - should warn (existing behavior)
// RUN: %clang_analyze_cc1 %s -verify=c11 -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

#include "Inputs/system-header-simulator.h"

extern char buf[128];
extern char src[128];

// c99-noflag-no-diagnostics

void test_memcpy(void) {
  memcpy(buf, src, 10);
  // c99-withflag-warning@-1{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard}}
  // c11-warning@-2{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memset(void) {
  memset(buf, 0, 10);
  // c99-withflag-warning@-1{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard}}
  // c11-warning@-2{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memmove(void) {
  memmove(buf, src, 10);
  // c99-withflag-warning@-1{{Call to function 'memmove' is insecure as it does not provide security checks introduced in the C11 standard}}
  // c11-warning@-2{{Call to function 'memmove' is insecure as it does not provide security checks introduced in the C11 standard}}
}

