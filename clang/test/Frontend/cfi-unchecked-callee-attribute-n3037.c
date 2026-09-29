// RUN: %clang_cc1 -Wall -Wno-unused -Wno-uninitialized -verify -std=c17 %s
// RUN: %clang_cc1 -Wall -Wno-unused -Wno-uninitialized -verify -std=c23 %s

#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

#if __STDC_VERSION__ >= 202311L
// expected-no-diagnostics
#endif

#if __STDC_VERSION__ < 202311L
// expected-note@+2 2 {{previous definition is here}}
#endif
struct field_attr_test {
  void (CFI_UNCHECKED_CALLEE *func)(void);
};

#if __STDC_VERSION__ < 202311L
// expected-error@+2{{redefinition of 'field_attr_test'}}
#endif
struct field_attr_test {
  void (CFI_UNCHECKED_CALLEE *func)(void);
};

typedef void (CFI_UNCHECKED_CALLEE func_t)(void);

#if __STDC_VERSION__ < 202311L
// expected-error@+2{{redefinition of 'field_attr_test'}}
#endif
struct field_attr_test {
  func_t *func;
};
