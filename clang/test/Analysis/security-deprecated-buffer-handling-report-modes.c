// DEFINE: %{analyze-cmd} = %clang_analyze_cc1 %s \
// DEFINE:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

// DEFINE: %{ReportMode} = -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode
// DEFINE: %{EnableAnnexK} = -D__STDC_LIB_EXT1__=200509L -D__STDC_WANT_LIB_EXT1__=1

// These cases should warn:
// RUN: %{analyze-cmd} -std=gnu99 %{ReportMode}=all                        -verify=common
// RUN: %{analyze-cmd} -std=gnu11                                          -verify=common
// RUN: %{analyze-cmd} -std=gnu11 %{ReportMode}=all                        -verify=common
// RUN: %{analyze-cmd} -std=gnu11 %{ReportMode}=c11-only                   -verify=common
// RUN: %{analyze-cmd} -std=gnu11 %{ReportMode}=actionable %{EnableAnnexK} -verify=common

// These cases should not warn:
// RUN: %{analyze-cmd} -std=gnu99                                          -verify=no-warning
// RUN: %{analyze-cmd} -std=gnu99 %{ReportMode}=actionable                 -verify=no-warning
// RUN: %{analyze-cmd} -std=gnu99 %{ReportMode}=c11-only                   -verify=no-warning
// RUN: %{analyze-cmd} -std=gnu11 %{ReportMode}=actionable                 -verify=no-warning

#include "Inputs/system-header-simulator.h"

extern char buf[128];
extern char src[128];

// no-warning-no-diagnostics

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
