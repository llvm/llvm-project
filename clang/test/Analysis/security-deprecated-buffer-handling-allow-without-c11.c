// These cases should not warn

// C99 with "all" mode
// RUN: %clang_analyze_cc1 %s -verify=common -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=all

// C11 with default mode
// RUN: %clang_analyze_cc1 %s -verify=common -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

// C11 with "all" mode
// RUN: %clang_analyze_cc1 %s -verify=common -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=all

// C11 with "c11-only" mode
// RUN: %clang_analyze_cc1 %s -verify=common -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=c11-only

// C11 with "actionable" mode and Annex K available
// RUN: %clang_analyze_cc1 %s -verify=common -std=gnu11 \
// RUN:   -D__STDC_LIB_EXT1__=200509L -D__STDC_WANT_LIB_EXT1__=1 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=actionable

// These cases should not warn

// C99 with default mode
// RUN: %clang_analyze_cc1 %s -verify=c99-default -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

// C99 with "actionable" mode and no Annex K
// RUN: %clang_analyze_cc1 %s -verify=c99-actionable -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=actionable

// C99 with "c11-only" mode
// RUN: %clang_analyze_cc1 %s -verify=c99-c11only -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=c11-only

// C11 with "actionable" mode and no Annex K
// RUN: %clang_analyze_cc1 %s -verify=c11-actionable-noannex -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling \
// RUN:   -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=actionable


#include "Inputs/system-header-simulator.h"

extern char buf[128];
extern char src[128];

// c99-default-no-diagnostics
// c99-actionable-no-diagnostics
// c99-c11only-no-diagnostics
// c11-actionable-noannex-no-diagnostics

void test_memcpy(void) {
  memcpy(buf, src, 10);
  // common-warning@-1{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memset(void) {
  memset(buf, 0, 10);
  // common-warning@-1{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memmove(void) {
  memmove(buf, src, 10);
  // common-warning@-1{{Call to function 'memmove' is insecure as it does not provide security checks introduced in the C11 standard}}
}

