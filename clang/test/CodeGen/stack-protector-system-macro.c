// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm-only -verify \
// RUN:   -stack-protector 3 -Wignored-attributes %s

// Test that 'stack_protector_ignore' attributes coming from system header
// macros don't trigger -Wignored-attributes under -fstack-protector-all.
// The attribute is not actionable at the expansion site: the user cannot
// remove it without editing the system header.

#ifdef IS_SYSHEADER
#pragma clang system_header

#define SANITY(a) (a / 0)

#define BUFFER_WITH_IGNORED_ATTR                                               \
  __extension__({                                                              \
    __attribute__((stack_protector_ignore)) char _buf[64];                     \
    _buf[0] = 0;                                                               \
  })

#else

#define IS_SYSHEADER
#include __FILE__

void testSanity(void) {
  // Validate that the test is set up correctly.
  int i = SANITY(0); // expected-warning {{division by zero is undefined}}
  (void)i;
}

void testSystemMacro(void) {
  // no -Wignored-attributes in system macro expansion
  BUFFER_WITH_IGNORED_ATTR;
}

void testUserCode(void) {
  // Written directly in user code, so it is still diagnosed.
  __attribute__((stack_protector_ignore))
  char buf[64]; // expected-warning {{'stack_protector_ignore' attribute ignored due to '-fstack-protector-all' option}}
  (void)buf;
}

#endif
