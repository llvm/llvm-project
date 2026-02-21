// RUN: %clang --target=i386 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// RUN: %clang --target=x86_64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"

// RUN: %clang --target=i386 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// RUN: %clang --target=x86_64 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF: "-mno-outline" "-mllvm" "-enable-machine-outliner=never"
