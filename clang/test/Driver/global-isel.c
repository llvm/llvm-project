// REQUIRES: x86-registered-target,aarch64-registered-target

// RUN: %clang -fexperimental-isel -S -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fno-experimental-isel -S -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// RUN: %clang -target aarch64 -fexperimental-isel -S %s -### 2>&1 | FileCheck --check-prefix=ARM64-DEFAULT %s
// RUN: %clang -target aarch64 -fexperimental-isel -S -O0 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O0 %s
// RUN: %clang -target aarch64 -fexperimental-isel -S -O2 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O2 %s
// RUN: %clang -target aarch64 -fexperimental-isel -Wno-experimental-isel -S -O2 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O2-NOWARN %s

// RUN: %clang -target x86_64 -fexperimental-isel -S %s -### 2>&1 | FileCheck --check-prefix=X86_64 %s

// ENABLED: "-mllvm" "-global-isel=1"
// DISABLED: "-mllvm" "-global-isel=0"

// ARM64-DEFAULT-NOT: warning: -fexperimental-sel
// ARM64-DEFAULT-NOT: "-global-isel-abort=2"
// ARM64-O0-NOT: warning: -fexperimental-sel
// ARM64-O2: warning: -fexperimental-isel support is incomplete for this architecture at the current optimization level
// ARM64-O2: "-mllvm" "-global-isel-abort=2"
// ARM64-O2-NOWARN-NOT: warning: -fexperimental-isel

// X86_64: -fexperimental-isel support for the 'x86_64' architecture is incomplete
// X86_64: "-mllvm" "-global-isel-abort=2"
