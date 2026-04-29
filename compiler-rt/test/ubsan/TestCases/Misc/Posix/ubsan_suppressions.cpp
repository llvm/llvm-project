// Test that we use the suppressions from __ubsan_default_suppressions.
// RUN: %clangxx -fsanitize=undefined %s -o %t && not %run %t 2>&1 | FileCheck %s

// Temporarily unsupporting on TySan until suppressions are implemented
// UNSUPPORTED: ubsan-tysan

extern "C" const char *__ubsan_default_suppressions() { return "FooBar"; }

// CHECK: {{.*}}Sanitizer: failed to parse suppressions
int main() {}
