// RUN: %clang -fsyntax-only -fdiagnostics-format=sarif %s -### 2>&1 | FileCheck %s --check-prefix=WARN
// WARN: warning: diagnostic formatting in SARIF mode is currently unstable [-Wsarif-format-unstable]

// RUN: %clang -fsyntax-only -fdiagnostics-format=SARIF %s -### 2>&1 | FileCheck %s --check-prefix=WARN2
// WARN2: warning: diagnostic formatting in SARIF mode is currently unstable [-Wsarif-format-unstable]
