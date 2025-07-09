// Test for the Triple constructor ambiguity fix on AIX
// This test verifies that the default target triple is correctly resolved
// and doesn't fall back to "unknown" due to constructor ambiguity.

// REQUIRES: system-aix, target={{.*}}-aix{{.*}}
// RUN: %clang -v %s -c 2>&1 | FileCheck %s --check-prefix=CHECK-TARGET

// Test that the target triple contains AIX and is not "unknown"
// The target should be something like "powerpc-ibm-aix7.3.0.0"
// CHECK-TARGET: Target: {{.*}}-aix{{.*}}

int main() {
    return 0;
}
