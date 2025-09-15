// Check that there is a warning when atos fails to symbolize an address
// and that atos continues symbolicating correctly after.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: iossim

void bar() {
  void *invalid_addr = reinterpret_cast<void *>(0xDEADBEEF);
  void (*func_ptr)() = reinterpret_cast<void (*)()>(invalid_addr);
  func_ptr();
}

int main() {
  bar();
  return 0;
  // CHECK: WARNING: atos failed to symbolize address{{.*}}
  // CHECK: {{.*}}atos-symbolized-recover.cpp:[[@LINE-3]]{{.*}}
}