/// Check that -ffixed register handled for globals.
/// Regression test for #76426, #109778
// REQUIRES: aarch64-registered-target

// RUN: %clang -c --target=aarch64-none-gnu -ffixed-x15 %s -o /dev/null 2>&1 | count 0

// RUN: not %clang -c --target=aarch64-none-gnu %s -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR_INVREG
// ERR_INVREG: error: register 'x15' unsuitable for global register variables on this target

// RUN: not %clang -c --target=aarch64-none-gnu -ffixed-x15 -DTYPE=short %s -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR_SIZE
// ERR_SIZE: error: size of register 'x15' does not match variable size

#ifndef TYPE
#define TYPE long
#endif

register TYPE x15 __asm__("x15");

TYPE foo() {
  return x15;
}
