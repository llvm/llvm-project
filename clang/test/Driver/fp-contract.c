// Test that -ffp-contract is set to the right value when combined with
// the options -ffast-math and -fno-fast-math.

// RUN: %clang -### -ffast-math -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// CHECK-FPC-FAST:     "-ffp-contract=fast"

// RUN: %clang -### -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s
// CHECK-FPC-ON:       "-ffp-contract=on"

// RUN: %clang -### -ffast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s
// CHECK-FPC-OFF:      "-ffp-contract=off"

// RUN: %clang -### -ffast-math -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=fast -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -ffp-contract=on -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -ffp-contract=off -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -ffp-contract=on -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s
// RUN: %clang -### -ffp-contract=off -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s


// RUN: %clang -### -ffast-math -ffp-contract=fast -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=on -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -ffast-math -ffp-contract=on -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffast-math -ffp-contract=off -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=off -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffast-math -ffp-contract=on -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=off -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -ffast-math -ffp-contract=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -fno-fast-math -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -fno-fast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -fno-fast-math -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=fast -fno-fast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffp-contract=fast -fno-fast-math -ffp-contract=off \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -ffp-contract=off -fno-fast-math -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=off -fno-fast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffp-contract=on -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=off -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=fast -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffp-contract=on -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffp-contract=off -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -ffp-contract=fast -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -fno-fast-math -ffast-math -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -fno-fast-math -ffast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -fno-fast-math -ffast-math -ffp-contract=off \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-OFF %s

