// Test for optional vendor additional distribution info in --version

// UNSUPPORTED: system-windows

// Check that "--verison" does not have extra output
// RUN: %clang --version  | FileCheck --check-prefix=CHECK-NO-DIST %s
// CHECK-NO-DIST-NOT: AMD AFAR drop

// Check that "--version" can have extra output
// RUN: echo "AMD AFAR drop" > /tmp/$$vers ; \
// RUN: export LLVM_VERSION_INFO_FILE=/tmp/$$vers; \
// RUN: %clang --version  | FileCheck --check-prefix=CHECK-DIST %s
// CHECK-DIST: Distribution: AMD AFAR drop
