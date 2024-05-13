// Test -fdirectives-only cases.

// We can manullay specify fdirectives-only, for any pre-processor job.
// RUN: %clang -### -std=c++20 -E -fdirectives-only foo.hh  2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-NON-HU %s

// Check that we automatically append -fdirectives-only for header-unit
// preprocessor jobs.
// RUN: %clang -### -std=c++20 -E -fmodule-header=user foo.hh  2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-HU %s

// CHECK-NON-HU: "-E"
// CHECK-NON-HU-SAME: "-fdirectives-only"
// CHECK-NON-HU-SAME: "-o" "-"
// CHECK-NON-HU-SAME: "-x" "c++-header" "foo.hh"

// CHECK-HU: "-E"
// CHECK-HU-SAME: "-fdirectives-only"
// CHECK-HU-SAME: "-o" "-"
// CHECK-HU-SAME: "-x" "c++-user-header" "foo.hh"
