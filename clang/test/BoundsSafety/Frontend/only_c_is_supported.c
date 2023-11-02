// RUN: not %clang -fexperimental-bounds-safety -x c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x objective-c %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x objective-c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x cuda -nocudalib -nocudainc %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang -fexperimental-bounds-safety -x renderscript %s 2>&1 | FileCheck -check-prefix ERR %s

// ERR: error: bounds safety is only supported for C

// This reports a warning to follow the default behavior of ClangAs.
// RUN: %clang -fexperimental-bounds-safety -x assembler -c %s -o /dev/null 2>&1 | FileCheck -check-prefix WARN %s

// '-x assembler-with-cpp' silently ignores unused options by default.
// Reporting a warning for -fbounds-safety when is used because preprocessor directives using the feature flag are currently not supported.
// specific warning instead of ignored?

// WARN: warning: argument unused during compilation: '-fexperimental-bounds-safety'

// expected-no-diagnostics
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x c %s -o /dev/null
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x assembler-with-cpp %s -o /dev/null

// '-x ir' test is done in a separate file because it doesn't recognize '//' as the comment prefix