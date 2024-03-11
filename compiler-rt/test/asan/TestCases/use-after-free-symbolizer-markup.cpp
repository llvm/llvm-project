/// End to end test for the sanitizer symbolizer markup.
/// Since it uses debug info to do offline symbolization we only check that the
/// current module is correctly symbolized.
// REQUIRES: linux
// RUN: rm -rf %t
// RUN: mkdir -p %t/.build-id/12
// RUN: %clangxx_asan %s -Wl,--build-id=0x12345678 -o %t/main
// RUN: cp %t/main %t/.build-id/12/345678.debug
// RUN: %env_asan_opts=enable_symbolizer_markup=1 not %run %t/main 2>%t/sanitizer.out
// RUN: llvm-symbolizer --filter-markup --debug-file-directory=%t < %t/sanitizer.out | FileCheck %s

#include <stdlib.h>

[[gnu::noinline]] char *alloc() {
  char *x = (char *)malloc(10 * sizeof(char));
  return x;
}
int main() {
  char *x = alloc();
  free(x);
  return x[5];
}
// CHECK: ERROR: AddressSanitizer: heap-use-after-free on address
// CHECK: {{0x.*}} at pc {{0x.*}} bp {{0x.*}} sp {{0x.*}}
// CHECK: READ of size 1 at {{0x.*}} thread T0
// CHECK:   #0 {{0x.*}} main{{.*}}use-after-free-symbolizer-markup.cpp:[[#@LINE - 5]]
// CHECK: {{0x.*}} is located 5 bytes inside of 10-byte region {{.0x.*,0x.*}}
// CHECK: freed by thread T0 here:
// CHECK:   #1 {{0x.*}} main{{.*}}use-after-free-symbolizer-markup.cpp:[[#@LINE - 9]]
// CHECK: previously allocated by thread T0 here:
// CHECK:   #1 {{0x.*}} alloc{{.*}}use-after-free-symbolizer-markup.cpp:[[#@LINE - 16]]
// CHECK:   #2 {{0x.*}} main {{.*}}use-after-free-symbolizer-markup.cpp:[[#@LINE - 13]]
// CHECK: Shadow byte legend (one shadow byte represents {{[0-9]+}} application bytes):
// CHECK: Global redzone:
// CHECK: ASan internal:
