// RUN: %clang -target armv7-linux-gnueabihf -moutline -c %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"
// RUN: %clang -target armv7-linux-gnueabihf -flto -moutline %s -### 2>&1 | FileCheck %s -check-prefix=ON-LTO
// ON-LTO: "-plugin-opt=-enable-machine-outliner"
// RUN: %clang -target armv7-linux-gnueabihf -moutline -mno-outline -c %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF: "-mllvm" "-enable-machine-outliner=never"
// RUN: %clang -target armv7-linux-gnueabihf -flto -moutline -mno-outline %s -### 2>&1 | FileCheck %s -check-prefix=OFF-LTO
// OFF-LTO: "-plugin-opt=-enable-machine-outliner=never"
