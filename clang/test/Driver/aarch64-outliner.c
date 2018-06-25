// REQUIRES: aarch64-registered-target
// RUN: %clang -target aarch64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"
// RUN: %clang -target aarch64 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF-NOT: "-mllvm" "-enable-machine-outliner"
// RUN: %clang -target aarch64 -moutline -flto=thin -S %s -### 2>&1 | FileCheck %s -check-prefix=FLTO
// FLTO: "-mllvm" "-enable-linkonceodr-outlining"
// RUN: %clang -target aarch64 -moutline -flto=full -S %s -### 2>&1 | FileCheck %s -check-prefix=TLTO
// TLTO: "-mllvm" "-enable-linkonceodr-outlining"
