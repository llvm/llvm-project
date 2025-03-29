// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=security.insecureAPI
// RUN: %clang_analyze_cc1 %s -verify -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI
// RUN: %clang_analyze_cc1 %s -verify -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI

#include "Inputs/system-header-simulator.h"

extern FILE *fp;
extern char buf[128];

void builtin_function_call_crash_fixes(char *c) {
  __builtin_strncpy(c, "", 6);
  __builtin_memset(c, '\0', (0));
  __builtin_memcpy(c, c, 0);
  __builtin_sprintf(buf, "%s", c);
  __builtin_fprintf(fp, "%s", c);

#if __STDC_VERSION__ > 199901
  // expected-warning@-7{{Call to function 'strncpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
  // expected-warning@-7{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard.}}
  // expected-warning@-7{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
  // expected-warning@-7{{Call to function 'sprintf' is insecure as it does not provide bounding of the memory buffer or security checks introduced in the C11 standard.}}
  // expected-warning@-7{{Call to function 'fprintf' is insecure as it does not provide bounding of the memory buffer or security checks introduced in the C11 standard.}}
#else
  // expected-no-diagnostics
#endif
}
