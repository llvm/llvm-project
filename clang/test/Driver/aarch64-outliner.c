// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// RUN: %clang --target=aarch64_be -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"
// RUN: %clang --target=aarch64 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// RUN: %clang --target=aarch64_be -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF: "-mllvm" "-enable-machine-outliner=never"
// RUN: %clang --target=x86_64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=WARN
// WARN: warning: 'x86_64' does not support '-moutline'; flag ignored [-Woption-ignored]
// WARN-NOT: "-mllvm" "-enable-machine-outliner"
