// ---------------------------------------------------------------------------
// Tests for the hvx qfloat target feature and backend flag if -mhvx-ieee-fp is
// passed on v79 and above.
// ---------------------------------------------------------------------------

// Test for v79, the correct backend flag is passed for -mhvx-ieee-fp.
// CHECK-IEEE: "-mllvm" "-hexagon-qfloat-mode=ieee"
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv79 -mhvx \
// RUN:   -mhvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// Test for arches lower than v79 does not pass any backend flag.
// CHECK-MODE-NOT: "-mllvm" "-hexagon-qfloat-mode="
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv75 -mhvx \
// RUN:   -mhvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-MODE %s

// Test for v79, the correct qfloat target feature is set when -mhvx-ieee-fp is
// passed.
// CHECK-HVX-QFLOAT-ON: "-target-feature" "+hvx-qfloat"
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv79 -mhvx \
// RUN:   -mhvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-HVX-QFLOAT-ON %s

// Test for arches lower than v79 does not set the qfloat target feature.
// CHECK-FEATURE-NOT: "-target-feature" "+hvx-qfloat"
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv75 -mhvx \
// RUN:   -mhvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-FEATURE %s
