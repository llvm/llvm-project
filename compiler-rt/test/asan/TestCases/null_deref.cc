// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

__attribute__((noinline))
// FIXME: Static symbols don't show up in PDBs. We can remove this once we start
// using DWARF.
#ifndef _MSC_VER
static
#endif
void NullDeref(int *ptr) {
  // CHECK: ERROR: AddressSanitizer: {{SEGV|access-violation}} on unknown address
  // CHECK:   {{0x0*000.. .*pc 0x.*}}
  ptr[10]++;  // BOOM
  // atos on Mac cannot extract the symbol name correctly. Also, on FreeBSD 9.2
  // the demangling function rejects local names with 'L' in front of them.
  // CHECK: {{    #0 0x.* in .*NullDeref.*null_deref.cc}}
}
int main() {
  NullDeref((int*)0);
  // CHECK: {{    #1 0x.* in main.*null_deref.cc}}
  // CHECK: AddressSanitizer can not provide additional info.
}
