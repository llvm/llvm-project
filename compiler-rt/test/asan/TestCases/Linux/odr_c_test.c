// Test that we can properly report an ODR violation between an instrumented
// global and a non-instrumented global if not using private aliases.

// RUN: %clang_asan -fcommon %s -fPIC -shared -mllvm -asan-use-private-alias=0 -o %dynamiclib1  -DFILE1
// RUN: %clang_asan -fcommon %s -fPIC -shared -mllvm -asan-use-private-alias=0 -o %dynamiclib2  -DFILE2
// RUN: %clang_asan -fcommon %s -fPIE %ld_flags_rpath_exe1 %ld_flags_rpath_exe2 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// RUN: %clang_asan -fcommon %s -fPIC -shared -mllvm -asan-use-private-alias=1 -o %dynamiclib1  -DFILE1
// RUN: %clang_asan -fcommon %s -fPIC -shared -mllvm -asan-use-private-alias=1 -o %dynamiclib2  -DFILE2
// RUN: %run %t 2>&1 | count 0

// Unaligned accesses don't work on strict-alignment targets like SPARC.
// UNSUPPORTED: sparc-target-arch

// CHECK: The following global variable is not properly aligned.
// CHECK: ERROR: AddressSanitizer: odr-violation
#if defined(FILE1)
__attribute__((aligned(8))) int x;
__attribute__((aligned(1))) char y;
// The gold linker puts ZZZ at the start of bss (where it is aligned)
// unless we have a large alternative like Displace:
__attribute__((aligned(1))) char Displace[105];
__attribute__((aligned(1))) char ZZZ[100];
#elif defined(FILE2)
int ZZZ = 1;
#else
extern int ZZZ;
int main() {
  return ZZZ;
}
#endif

