/// Test instrumentation can handle various linkages.
// RUN: %clang_profgen -fcoverage-mapping %s -o %t
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t

// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -Wl,--gc-sections %s -o %t
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t

#include <stdio.h>

void discarded0() {}
__attribute__((weak)) void discarded1() {}

void external() { puts("external"); }
__attribute__((weak)) void weak() { puts("weak"); }
static void internal() { puts("internal"); }
__attribute__((noinline)) inline void linkonce_odr() { puts("linkonce_odr"); }

int main() {
  internal();
  external();
  weak();
  linkonce_odr();
}
