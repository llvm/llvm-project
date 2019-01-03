// RUN: %clang -target i686-windows-msvc -c %s -### 2>&1 | FileCheck -check-prefix=MSVC %s
// RUN: %clang -target x86_64-windows-msvc -c %s -### 2>&1 | FileCheck -check-prefix=MSVC %s
// RUN: %clang -target i686-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-DWARF %s
// RUN: %clang -target x86_64-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-SEH %s
// RUN: %clang -target aarch64-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-DWARF %s
// RUN: %clang -target aarch64-windows-gnu -fseh-exceptions -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-SEH %s

MSVC-NOT: -fdwarf-exceptions
MSVC-NOT: -fseh-exceptions
MINGW-DWARF: -fdwarf-exceptions
MINGW-SEH: -munwind-tables
MINGW-SEH: -fseh-exceptions
