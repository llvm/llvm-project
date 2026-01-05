// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify %s
// expected-no-diagnostics

int memcmp(const void *, const void *, unsigned long);

__typeof(memcmp) memcmp_alias __asm__("memory_compare") __attribute__((noreturn));

void use(void) {
  (void)memcmp_alias(0, 0, 0);
}

// Also test a function-type typedef rather than __typeof
typedef int memcmp_type(const void *, const void *, unsigned long);

memcmp_type memcmp_alias2 __asm__("memory_compare2") __attribute__((noreturn));

void use2(void) {
  (void)memcmp_alias2(0, 0, 0);
}
