// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=limited -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=smith -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SMITH %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=extend -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=EXTND %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=limited \
// RUN: -fcomplex-arithmetic=smith -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=limited \
// RUN: -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN2 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=limited \
// RUN: -fcomplex-arithmetic=extend -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN3 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=smith \
// RUN: -fcomplex-arithmetic=limited  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN4 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=smith \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN5 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=smith \
// RUN: -fcomplex-arithmetic=extend  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN6 %s


// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=extend \
// RUN: -fcomplex-arithmetic=limited  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN7 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=extend \
// RUN: -fcomplex-arithmetic=smith  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN8 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=extend \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN9 %s


// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=limited  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN10 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=smith  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN11 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=extend  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN12 %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -ffast-math -fcomplex-arithmetic=limited -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=limited -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LMTD %s

// LMTD: -complex-range=limited
// FULL: -complex-range=full
// EXTND: -complex-range=extend
// LMTD-NOT: -complex-range=smith
// CHECK-NOT: -complex-range=limited
// SMITH: -complex-range=smith
// SMITH-NOT: -complex-range=limited
// CHECK-NOT: -complex-range=smith
// WARN1: warning: overriding '-fcomplex-arithmetic=limited' option with '-fcomplex-arithmetic=smith' [-Woverriding-option]
// WARN2: warning: overriding '-fcomplex-arithmetic=limited' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN3: warning: overriding '-fcomplex-arithmetic=limited' option with '-fcomplex-arithmetic=extend' [-Woverriding-option]
// WARN4: warning: overriding '-fcomplex-arithmetic=smith' option with '-fcomplex-arithmetic=limited' [-Woverriding-option]
// WARN5: warning: overriding '-fcomplex-arithmetic=smith' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN6: warning: overriding '-fcomplex-arithmetic=smith' option with '-fcomplex-arithmetic=extend' [-Woverriding-option]
// WARN7: warning: overriding '-fcomplex-arithmetic=extend' option with '-fcomplex-arithmetic=limited' [-Woverriding-option]
// WARN8: warning: overriding '-fcomplex-arithmetic=extend' option with '-fcomplex-arithmetic=smith' [-Woverriding-option]
// WARN9: warning: overriding '-fcomplex-arithmetic=extend' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN10: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=limited' [-Woverriding-option]
// WARN11: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=smith' [-Woverriding-option]
// WARN12: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=extend' [-Woverriding-option]
