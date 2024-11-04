// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FRTRN %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck  %s

// RUN: %clang -### -target x86_64 -fcx-limited-range \
// RUN: -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules \
// RUN: -fcx-limited-range  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN2 %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -ffast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -ffast-math -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -Werror -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -Werror -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FRTRN %s

// LMTD: -complex-range=limited
// FULL: -complex-range=full
// LMTD-NOT: -complex-range=fortran
// CHECK-NOT: -complex-range=limited
// FRTRN: -complex-range=fortran
// FRTRN-NOT: -complex-range=limited
// CHECK-NOT: -complex-range=fortran
// WARN1: warning: overriding '-fcx-limited-range' option with '-fcx-fortran-rules' [-Woverriding-option]
// WARN2: warning: overriding '-fcx-fortran-rules' option with '-fcx-limited-range' [-Woverriding-option]
