// RUN: %clang --target=x86_64-windows-gnu -c -mwindows %s -### 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-windows-gnu -c -mconsole %s -### 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-windows-gnu -c -mdll %s -### 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-windows-gnu -c -mthreads %s -### 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: not %clang --target=x86_64-windows-msvc -c -mwindows %s -### 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang --target=x86_64-windows-msvc -c -mconsole %s -### 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang --target=x86_64-windows-msvc -c -mdll %s -### 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang --target=x86_64-windows-msvc -c -mthreads %s -### 2>&1 | FileCheck %s --check-prefix=ERROR
// WARNING: warning: argument unused during compilation: '{{.*}}' [-Wunused-command-line-argument]
// ERROR: error: unsupported option '{{.*}}' for target '{{.*}}'
