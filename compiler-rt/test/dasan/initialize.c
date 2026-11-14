// RUN: %clang_dasan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

int main(void) { return 0; }

// CHECK: DeviceAddressSanitizer: initialized
