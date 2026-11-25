// RUN: %clangxx_csan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --allow-empty

using Vec16 = int __attribute__((vector_size(16)));
using Vec32 = int __attribute__((vector_size(32)));

volatile Vec16 Global16;
volatile Vec32 Global32;

int main() {
  Vec16 V16 = {};
  Vec32 V32 = {};
  for (int I = 0; I < 1024; ++I) {
    Global16 = V16;
    Global32 = V32;
  }
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
