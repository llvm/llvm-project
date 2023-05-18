/// Test instrumentation can handle various linkages.
// REQUIRES: lld-available

// FIXME: Investigate and fix.
// XFAIL: powerpc64-target-arch

// RUN: %clang_profgen -fcoverage-mapping %s -o %t
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s

// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -fuse-ld=lld -Wl,--gc-sections %s -o %t
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s

// CHECK:      {{.*}}external{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1
// CHECK:      {{.*}}weak{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1
// CHECK:      main:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1
// CHECK:      {{.*}}internal{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1
// CHECK:      {{.*}}linkonce_odr{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1

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
