// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv8 -target-feature +neon %s -S -o /dev/null -verify -DWARN
// RUN: %clang_cc1 -triple armv8 -target-feature +neon %s -S -o /dev/null -Werror -verify

void set_endian() {
  asm("setend be");
// expected-note@1 {{instantiated into assembly here}}
#ifdef WARN
// expected-warning@-3 {{deprecated}}
#else
// expected-error@-5 {{deprecated}}
#endif
}
