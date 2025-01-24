// Verify that PLT optimization in BOLT preserves exception-handling info.

// REQUIRES: system-linux

// RUN: %clangxx %cxxflags -O1 -Wl,-q,-znow %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt.exe --plt=all --print-only=.*main.* \
// RUN:   --print-finalized 2>&1 | FileCheck %s

// CHECK-LABEL: Binary Function
// CHECK:      adrp {{.*}}__cxa_throw
// CHECK-NEXT: ldr {{.*}}__cxa_throw
// CHECK-NEXT: blr x17 {{.*}} handler: {{.*}} PLTCall:

int main() {
  try {
    throw new int;
  } catch (...) {
    return 0;
  }
  return 1;
}
