// RUN: %clangxx_csan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --allow-empty

int Global;

int main() {
  for (int I = 0; I < 1024; ++I)
    Global++;
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
