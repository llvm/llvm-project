// Default: nothing is passed to cc1.
// RUN: %clang -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT-NOT: "-fno-diagnostics-show-note-snippets"
// DEFAULT-NOT: "-fdiagnostics-show-note-snippets"

// RUN: %clang -### -fsyntax-only -fno-diagnostics-show-note-snippets %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLED
// DISABLED: "-fno-diagnostics-show-note-snippets"

// RUN: %clang -### -fsyntax-only -fno-diagnostics-show-note-snippets \
// RUN:   -fdiagnostics-show-note-snippets %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
