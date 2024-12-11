

// =============================================================================
// Adoption mode off
// =============================================================================

// Check adoption mode is off by default
// RUN: %clang -fbounds-safety -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-ADOPT-CHECK %s

// Check adoption mode is off if explicitly requested
// RUN: %clang -fbounds-safety -fno-bounds-safety-adoption-mode -c %s -### \
// RUN: 2>&1 | FileCheck --check-prefix=NO-ADOPT-CHECK %s

// NO-ADOPT-CHECK: -fbounds-safety
// NOT-ADOPT-CHECK-NOT: -fbounds-safety-adoption-mode

// =============================================================================
// Adoption mode on
// =============================================================================

// Check driver passes flag when adoption mode explicitly requested
// RUN: %clang -fbounds-safety -fbounds-safety-adoption-mode -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=ADOPT-CHECK %s

// Check -fno-<flag> followed by -f<flag> causes -f<flag> to win
// RUN: %clang -fbounds-safety -fno-bounds-safety-adoption-mode \
// RUN: -fbounds-safety-adoption-mode -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=ADOPT-CHECK %s

// ADOPT-CHECK: -fbounds-safety
// ADOPT-CHECK: -fbounds-safety-adoption-mode
