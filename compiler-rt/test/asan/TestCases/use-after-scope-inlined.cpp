// Test with "-O2" only to make sure inlining (leading to use-after-scope)
// happens. "always_inline" is not enough, as Clang doesn't emit
// llvm.lifetime intrinsics at -O0.
//
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

int *arr;

__attribute__((always_inline))
void inlined(int arg) {
  int x[5];
  for (int i = 0; i < arg; i++) x[i] = i;
  arr = x;
}

int main(int argc, char *argv[]) {
  inlined(argc);
  return arr[argc - 1];  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK: READ of size 4 at 0x{{.*}} thread T0
  // CHECK:   #0 0x{{.*}} in main
  // CHECK:      {{.*}}use-after-scope-inlined.cpp:[[@LINE-4]]
  // CHECK: Address 0x{{.*}} is located in stack of thread T0 at offset [[OFFSET:[^ ]*]] in frame
  // CHECK:      {{.*}} in main
  // CHECK:   This frame has
  // CHECK:     {{\[}}[[OFFSET]], {{.*}}) 'x.i' (line [[@LINE-15]])
}
