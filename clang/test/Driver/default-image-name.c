// RUN: %clang -target i386-unknown-linux-gnu %s -### 2>&1 | FileCheck -check-prefix=LINUX %s
// LINUX: a.out

// RUN: %clang -target i686-pc-windows-msvc %s -### 2>&1 | FileCheck -check-prefix=WIN %s
// RUN: %clang -target i686-pc-windows-gnu %s -### 2>&1 | FileCheck -check-prefix=WIN %s
// RUN: %clang -target i686-windows-gnu %s -### 2>&1 | FileCheck -check-prefix=WIN %s
// WIN: a.exe
