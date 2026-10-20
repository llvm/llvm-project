/// Test -msimd options.

/// COM: -msimd=none
// RUN: %clang --target=loongarch64 -mlasx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX
// RUN: %clang --target=loongarch64 -mlasx -mlsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX

// RUN: %clang --target=loongarch64 -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mlsx -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlsx -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX

// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mlsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mlsx -msimd=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX


/// COM: -msimd=lsx
// RUN: %clang --target=loongarch64 -mlasx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX
// RUN: %clang --target=loongarch64 -mlasx -mlsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX

// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlsx -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mlsx -mno-lsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NOLSX,NOLASX

// RUN: %clang --target=loongarch64 -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mlsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX


/// COM: -msimd=lasx
// RUN: %clang --target=loongarch64 -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX
// RUN: %clang --target=loongarch64 -mlasx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX
// RUN: %clang --target=loongarch64 -mlasx -mlsx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX
// RUN: %clang --target=loongarch64 -mlsx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,LASX

// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX

// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mlsx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -msimd=lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -mlsx -msimd=lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=LSX,NOLASX


// NOLSX-NOT: "-target-feature" "+lsx"
// NOLASX-NOT: "-target-feature" "+lasx"
// LSX-DAG: "-target-feature" "+lsx"
// LASX-DAG: "-target-feature" "+lasx"
// NOLSX-NOT: "-target-feature" "+lsx"
// NOLASX-NOT: "-target-feature" "+lasx"
