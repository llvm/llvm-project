// Check that basic use of win32-macho targets works.
// RUN: %clang -fsyntax-only -target x86_64-pc-win32-macho %s

// RUN: %clang -fsyntax-only -target x86_64-pc-win32-macho -g %s -### 2>&1 | FileCheck %s -check-prefix=DEBUG-INFO
// DEBUG-INFO: -dwarf-version={{.*}}
