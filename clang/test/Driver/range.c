// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=PRMTD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN2 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN3 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN4 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN5 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=promoted  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN6 %s


// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN7 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=improved  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN8 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN9 %s


// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN10 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=improved  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN11 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=promoted  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN12 %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// BASIC: -complex-range=basic
// FULL: -complex-range=full
// PRMTD: -complex-range=promoted
// BASIC-NOT: -complex-range=improved
// CHECK-NOT: -complex-range=basic
// IMPRVD: -complex-range=improved
// IMPRVD-NOT: -complex-range=basic
// CHECK-NOT: -complex-range=improved
// WARN1: warning: overriding '-fcomplex-arithmetic=basic' option with '-fcomplex-arithmetic=improved' [-Woverriding-option]
// WARN2: warning: overriding '-fcomplex-arithmetic=basic' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN3: warning: overriding '-fcomplex-arithmetic=basic' option with '-fcomplex-arithmetic=promoted' [-Woverriding-option]
// WARN4: warning: overriding '-fcomplex-arithmetic=improved' option with '-fcomplex-arithmetic=basic' [-Woverriding-option]
// WARN5: warning: overriding '-fcomplex-arithmetic=improved' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN6: warning: overriding '-fcomplex-arithmetic=improved' option with '-fcomplex-arithmetic=promoted' [-Woverriding-option]
// WARN7: warning: overriding '-fcomplex-arithmetic=promoted' option with '-fcomplex-arithmetic=basic' [-Woverriding-option]
// WARN8: warning: overriding '-fcomplex-arithmetic=promoted' option with '-fcomplex-arithmetic=improved' [-Woverriding-option]
// WARN9: warning: overriding '-fcomplex-arithmetic=promoted' option with '-fcomplex-arithmetic=full' [-Woverriding-option]
// WARN10: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=basic' [-Woverriding-option]
// WARN11: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=improved' [-Woverriding-option]
// WARN12: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=promoted' [-Woverriding-option]
