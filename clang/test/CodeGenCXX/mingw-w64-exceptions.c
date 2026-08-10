// RUN: %clang -target x86_64-windows-gnu -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SEH
// RUN: %clang -target i686-windows-gnu -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DWARF

// RUN: %clang -target x86_64-windows-gnu -fsjlj-exceptions -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-SJLJ

// RUN: %clang -target x86_64-windows-gnu -fdwarf-exceptions -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-DWARF

// RUN: %clang -target x86_64-windows-gnu -fsjlj-exceptions -fseh-exceptions -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-SEH

// RUN: %clang -target x86_64-windows-gnu -fseh-exceptions -fsjlj-exceptions -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-SJLJ

// RUN: %clang -target x86_64-windows-gnu -fseh-exceptions -fdwarf-exceptions -c %s -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-DWARF

// CHECK-SEH: "-exception-model=seh"
// CHECK-SJLJ: "-exception-model=sjlj"
// CHECK-DWARF-NOT: "-exception-model=sjlj"
// CHECK-DWARF-NOT: "-exception-model=seh"
