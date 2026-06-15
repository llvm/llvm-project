// RUN: %clang --target=ppc64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=WARN
// WARN: warning: 'ppc64' does not support '-moutline'; flag ignored [-Woption-ignored]
// WARN-NOT: "-mllvm" "-enable-machine-outliner"
