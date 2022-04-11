// RUN: %clang --target=x86_64-windows -c -Wa,-mbig-obj %s -### 2>&1 | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang --target=x86_64-windows -c -Xassembler -mbig-obj %s -### 2>&1 | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang --target=x86_64-linux -c -Wa,-mbig-obj %s -### 2>&1 | FileCheck %s --check-prefix=LINUX
// RUN: %clang --target=x86_64-linux -c -Xassembler -mbig-obj %s -### 2>&1 | FileCheck %s --check-prefix=LINUX
// WINDOWS-NOT: argument unused during compilation
// LINUX: unsupported argument '-mbig-obj' to option '-{{(Wa,|Xassembler)}}'
