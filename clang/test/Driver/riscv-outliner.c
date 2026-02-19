// RUN: %clang --target=riscv32 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// RUN: %clang --target=riscv64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"

// RUN: %clang --target=riscv32 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// RUN: %clang --target=riscv64 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF: "-mno-outline" "-mllvm" "-enable-machine-outliner=never"
