// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FRTRN %s

// RUN: %clang -### -target x86_64 -fcx-limited-range \
// RUN: -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules \
// RUN: -fcx-limited-range  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN2 %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FRTRN %s

// RUN: %clang -### -target x86_64 -ffast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FRTRN %s

// LMTD: -complex-range=cx_limited
// FRTRN: -complex-range=cx_fortran
// WARN1: warning: overriding '-fcx-limited-range' option with '-fcx-fortran-rules' [-Woverriding-option]
// WARN2: warning: overriding '-fcx-fortran-rules' option with '-fcx-limited-range' [-Woverriding-option]
