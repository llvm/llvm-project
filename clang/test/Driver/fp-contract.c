// Test that -ffp-contract is set to the right value when combined with
// the options -ffast-math, -fno-fast-math, funsafe-math-optimizations,
// fno-unsafe-math-optimizations.

// These warning checks are above the run lines because the warning is reported
// before the drive options that are checked below the run lines.
// WARN_FM_OFF: warning: overriding '-ffast-math' option with '-ffp-contract=off'
// WARN_FM_ON: warning: overriding '-ffast-math' option with '-ffp-contract=on'
// WARN_FM_FHP: warning: overriding '-ffast-math' option with '-ffp-contract=fast-honor-pragmas'
// WARN_UM_OFF: warning: overriding '-funsafe-math-optimizations' option with '-ffp-contract=off'
// WARN_UM_ON: warning: overriding '-funsafe-math-optimizations' option with '-ffp-contract=on'

// ffast-math, fno-fast-math
// RUN: %clang -### -ffast-math -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// CHECK-FPC-FAST:     "-ffp-contract=fast"

// RUN: %clang -### -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_FM_ON %s
// CHECK-FPC-ON:       "-ffp-contract=on"

// RUN: %clang -### -ffast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_FM_OFF %s
// CHECK-FPC-OFF:      "-ffp-contract=off"

// RUN: %clang -### -Werror -ffast-math -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -ffast-math -ffp-contract=fast-honor-pragmas -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-FAST-HONOR,WARN_FM_FHP %s
// CHECK-FPC-FAST-HONOR:     "-ffp-contract=fast-honor-pragmas"

// RUN: %clang -### -Werror -ffp-contract=fast -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=on -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=off -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=on -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s
// RUN: %clang -### -Werror -ffp-contract=off -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s


// RUN: %clang -### -Werror -ffast-math -ffp-contract=fast -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -ffast-math -ffp-contract=on -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_FM_ON %s

// RUN: %clang -### -ffast-math -ffp-contract=on -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-FAST,WARN_FM_ON %s

// RUN: %clang -### -ffast-math -ffp-contract=off -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_FM_OFF %s

// RUN: %clang -### -ffast-math -ffp-contract=off -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-FAST,WARN_FM_OFF %s

// RUN: %clang -### -ffast-math -ffp-contract=on -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_FM_ON %s

// RUN: %clang -### -ffast-math -ffp-contract=off -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_FM_OFF %s

// RUN: %clang -### -Werror -ffast-math -ffp-contract=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-fast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-fast-math -ffp-contract=off \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -ffp-contract=off -fno-fast-math -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=off -fno-fast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=on -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=off -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=on -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=off -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -ffp-contract=fast -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -fno-fast-math -ffast-math -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -fno-fast-math -ffast-math -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_FM_ON %s

// RUN: %clang -### -fno-fast-math -ffast-math -ffp-contract=off \
// RUN: -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_FM_OFF %s

// funsafe-math-optimizations, fno-unsafe-math-optimizations
// RUN: %clang -### -Werror -funsafe-math-optimizations -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_UM_ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_UM_OFF %s

// RUN: %clang -### -Werror -funsafe-math-optimizations -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=on -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=off -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-unsafe-math-optimizations -c \
// RUN: %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s
// RUN: %clang -### -Werror -ffp-contract=on -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s
// RUN: %clang -### -Werror -ffp-contract=off -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -funsafe-math-optimizations -ffp-contract=fast \
// RUN: -ffp-contract=on -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=on \
// RUN: -ffp-contract=off -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_UM_ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=on \
// RUN: -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-FAST,WARN_UM_ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=off \
// RUN: -ffp-contract=on -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_UM_OFF %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=off \
// RUN: -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-FAST,WARN_UM_OFF %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=on \
// RUN: -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_UM_ON %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=off \
// RUN: -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_UM_OFF %s

// RUN: %clang -### -Werror -funsafe-math-optimizations -ffp-contract=fast \
// RUN: -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -funsafe-math-optimizations -fno-unsafe-math-optimizations \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -funsafe-math-optimizations \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-unsafe-math-optimizations \
// RUN: -ffp-contract=on \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=fast -fno-unsafe-math-optimizations \
// RUN: -ffp-contract=off \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -ffp-contract=off -fno-unsafe-math-optimizations \
// RUN: -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=off -fno-unsafe-math-optimizations \
// RUN: -ffp-contract=on -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=on -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=off -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=fast -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -ffp-contract=on -funsafe-math-optimizations -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-ON %s

// RUN: %clang -### -Werror -ffp-contract=off -funsafe-math-optimizations -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-OFF %s

// RUN: %clang -### -Werror -ffp-contract=fast -funsafe-math-optimizations -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -Werror -fno-unsafe-math-optimizations -funsafe-math-optimizations \
// RUN:   -ffp-contract=fast \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=CHECK-FPC-FAST %s

// RUN: %clang -### -fno-unsafe-math-optimizations -funsafe-math-optimizations \
// RUN:   -ffp-contract=on -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-ON,WARN_UM_ON %s

// RUN: %clang -### -fno-unsafe-math-optimizations -funsafe-math-optimizations \
// RUN:   -ffp-contract=off -c %s 2>&1 | FileCheck --check-prefixes=CHECK-FPC-OFF,WARN_UM_OFF %s

// RUN: %clang -### -funsafe-math-optimizations -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN_UM_OFF %s

// This case should not warn
// RUN: %clang -### -Werror -funsafe-math-optimizations \
// RUN: -fno-unsafe-math-optimizations -ffp-contract=off -c %s

// RUN: %clang -### -ffast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN_FM_OFF %s

// This case should not warn
// RUN: %clang -### -Werror -ffast-math -fno-fast-math -ffp-contract=off -c %s
