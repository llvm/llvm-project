// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=security.insecureAPI
// RUN: %clang_analyze_cc1 %s -verify -std=gnu11 \
// RUN:   -analyzer-checker=security.insecureAPI
// RUN: %clang_analyze_cc1 %s -verify -std=gnu99 \
// RUN:   -analyzer-checker=security.insecureAPI

void builtin_function_call_crash_fixes(char *c) {
  __builtin_strncpy(c, "", 6);
  __builtin_memset(c, '\0', (0));
  __builtin_memcpy(c, c, 0);

#if __STDC_VERSION__ > 199901
  // expected-warning@-5{{Call to function 'strncpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
  // expected-warning@-5{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard.}}
  // expected-warning@-5{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
#else
  // expected-no-diagnostics
#endif
}
