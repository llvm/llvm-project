// REQUIRES: powerpc-registered-target
// RUN: %clang -target powerpc64le -c %s -mllvm -stop-after=finalize-isel -o - | \
// RUN:   FileCheck %s
// RUN: %clang -target powerpc64 -c %s -mllvm -stop-after=finalize-isel -o - | \
// RUN:   FileCheck %s

void test_function(void) {
  asm volatile("":::"ca");
  asm volatile("":::"xer");
  // CHECK: call void asm sideeffect "", "~{xer}"()
  // CHECK: call void asm sideeffect "", "~{xer}"()
  // CHECK: INLINEASM &"", {{.*}} implicit-def early-clobber $xer
  // CHECK: INLINEASM &"", {{.*}} implicit-def early-clobber $xer
}
