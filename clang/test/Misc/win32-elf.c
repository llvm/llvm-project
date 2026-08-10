// Check that basic use of win32-elf targets works.
// RUN: %clang -fsyntax-only -target x86_64-pc-win32-elf %s

// RUN: %clang -fsyntax-only -target x86_64-pc-win32-elf -g %s -### 2>&1 | FileCheck %s -check-prefix=DEBUG-INFO
// DEBUG-INFO: -dwarf-version={{.*}}
