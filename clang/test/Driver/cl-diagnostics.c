// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /diagnostics:classic -### -- %s 2>&1 | FileCheck %s --check-prefix=CLASSIC
// CLASSIC: -fno-caret-diagnostics
// CLASSIC: -fno-show-column

// RUN: %clang_cl /diagnostics:column -### -- %s 2>&1 | FileCheck %s --check-prefix=COLUMN
// COLUMN: -fno-caret-diagnostics
// COLUMN-NOT: -fno-show-column

// RUN: %clang_cl /diagnostics:caret -### -- %s 2>&1 | FileCheck %s --check-prefix=CARET
// CARET-NOT: -fno-caret-diagnostics
// CARET-NOT: -fno-show-column

// RUN: not %clang_cl -fms-compatibility-version=19  /diagnostics:classic /Zs -c -- %s 2>&1 | FileCheck %s --check-prefix=OUTPUT_CLASSIC

// OUTPUT_CLASSIC: cl-diagnostics.c({{[0-9]+}}): error: "asdf"
// OUTPUT_CLASSIC-NOT: #error

// RUN: not %clang_cl -fms-compatibility-version=19 /diagnostics:caret /Zs -c -- %s 2>&1 | FileCheck %s --check-prefix=OUTPUT_CARET

// OUTPUT_CARET: cl-diagnostics.c({{[0-9]+,[0-9]+}}): error: "asdf"
// OUTPUT_CARET-NEXT: #error "asdf"
// OUTPUT_CARET-NEXT: ^


#error "asdf"
