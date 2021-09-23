// RUN: %clang -target i686-windows-msvc -c %s -### 2>&1 | FileCheck -check-prefix=MSVC %s
// RUN: %clang -target x86_64-windows-msvc -c %s -### 2>&1 | FileCheck -check-prefix=MSVC %s
// RUN: %clang -target i686-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-DWARF %s
// RUN: %clang -target x86_64-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-SEH %s
// RUN: %clang -target aarch64-windows-gnu -fdwarf-exceptions -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-DWARF %s
// RUN: %clang -target aarch64-windows-gnu -c %s -### 2>&1 | FileCheck -check-prefix=MINGW-SEH %s

MSVC-NOT: -exception-model=dwarf
MSVC-NOT: -exception-model=seh
MINGW-DWARF: -exception-model=dwarf
MINGW-SEH: -funwind-tables=2
MINGW-SEH: -exception-model=seh
