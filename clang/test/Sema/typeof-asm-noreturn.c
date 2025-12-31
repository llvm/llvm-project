// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify %s
// expected-no-diagnostics

int memcmp(const void *, const void *, unsigned long);

__typeof(memcmp) memcmp_alias __asm__("memory_compare") __attribute__((noreturn));

void use(void) {
  (void)memcmp_alias(0, 0, 0);
}
