// RUN: clang-include-cleaner %s -- -Wunused 2>&1 | FileCheck --allow-empty %s
static void foo() {}

// Make sure that we don't get an unused warning
// CHECK-NOT: unused function
