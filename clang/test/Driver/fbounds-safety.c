// RUN: %clang -c %s -### 2>&1 | FileCheck -check-prefix NOFLAG %s
// RUN: %clang -fexperimental-bounds-safety -fno-experimental-bounds-safety -c %s -### 2>&1 | FileCheck -check-prefix NOFLAG %s
// NOFLAG-NOT: -fexperimental-bounds-safety

// RUN: %clang -fexperimental-bounds-safety -### %s 2>&1 | FileCheck -check-prefix FLAG %s
// RUN: %clang -fno-experimental-bounds-safety -fexperimental-bounds-safety -c %s -### 2>&1 | FileCheck -check-prefix FLAG %s
// FLAG: -fexperimental-bounds-safety

// RUN: not %clang -fexperimental-bounds-safety -x c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x objective-c %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x objective-c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x cuda -nocudalib -nocudainc %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x renderscript %s 2>&1 | FileCheck -check-prefix ERR %s
// ERR: error: '-fexperimental-bounds-safety' is only supported for C

// This reports a warning to follow the default behavior of ClangAs.
// RUN: %clang -fexperimental-bounds-safety -x assembler -c %s -o /dev/null 2>&1 | FileCheck -check-prefix WARN %s
// WARN: warning: argument unused during compilation: '-fexperimental-bounds-safety'

// expected-no-diagnostics
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x c %s -o /dev/null
// Unlike '-x assembler', '-x assembler-with-cpp' silently ignores unused options by default.
// XXX: Should report a targeted warning in future when assembler tries to use preprocessor directives to conditionalize behavior when bounds safety is enabled.
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x assembler-with-cpp %s -o /dev/null
// RUN: %clang -### -x ir -fexperimental-bounds-safety %s -Xclang -verify -o /dev/null
