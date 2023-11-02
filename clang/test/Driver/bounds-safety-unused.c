// This reports a warning to follow the default behavior of ClangAs.
// RUN: %clang -fexperimental-bounds-safety -x assembler -c %s -o /dev/null 2>&1 | FileCheck -check-prefix WARN %s


// WARN: warning: argument unused during compilation: '-fexperimental-bounds-safety'

// expected-no-diagnostics
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x c %s -o /dev/null
// Unlike '-x assembler', '-x assembler-with-cpp' silently ignores unused options by default.
// XXX: Should report a targeted warning in future when assembler tries to use preprocessor directives to conditionalize behavior when bounds safety is enabled.
// RUN: %clang -fexperimental-bounds-safety -Xclang -verify -c -x assembler-with-cpp %s -o /dev/null
// RUN: %clang -### -x ir -fexperimental-bounds-safety %s -Xclang -verify -o /dev/null
