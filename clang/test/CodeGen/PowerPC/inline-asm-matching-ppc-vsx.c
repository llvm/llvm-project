// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature +vsx \
// RUN:   -target-cpu pwr9 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -target-feature +vsx \
// RUN:   -target-cpu pwr9 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -target-feature +vsx \
// RUN:   -target-cpu pwr9 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix -target-feature +vsx \
// RUN:   -target-cpu pwr9 -emit-llvm %s -o - | FileCheck %s

// This case is to test VSX register support in the clobbers list for inline asm.
void testVSX (void) {
  unsigned int a = 0;
  unsigned int *dbell=&a;
  int d;
  __asm__ __volatile__ (
    "lxvw4x  32, 0, %2\n\t"
    "stxvw4x 32, 0, %1"
    : "=m"(*(volatile unsigned int*)(dbell))
    : "r" (dbell), "r" (&d)
    : "vs32"
  );
}

// CHECK: call void asm sideeffect "lxvw4x  32, 0, $2\0A\09stxvw4x 32, 0, $1", "=*m,r,r,~{vs32}"
