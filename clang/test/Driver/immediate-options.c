// RUN: %clang --help | FileCheck %s -check-prefix=HELP
// HELP: isystem
// HELP-NOT: ast-dump
// HELP-NOT: driver-mode

// Make sure that Flang-only options are not available in Clang
// HELP-NOT: test-io

// RUN: %clang --help-hidden | FileCheck %s -check-prefix=HELP-HIDDEN
// HELP-HIDDEN: driver-mode
// HELP-HIDDEN-NOT: test-io

// RUN: %clang -dumpversion | FileCheck %s -check-prefix=DUMPVERSION
// DUMPVERSION: {{[0-9]+\.[0-9.]+}}

// Test if the -print-resource-dir option is accepted without error.
// Allow unspecified output because the value of CLANG_RESOURCE_DIR is unknown.
// RUN: %clang -print-resource-dir | FileCheck %s -check-prefix=PRINT-RESOURCE-DIR
// PRINT-RESOURCE-DIR: {{.+}}

// Default resource-dir layout
// RUN: %clang -print-runtime-dir --target=x86_64-pc-windows-msvc \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s
// PRINT-RUNTIME-DIR: lib{{/|\\}}windows

// Per target dir layout
// RUN: %clang -print-runtime-dir --target=x86_64-pc-windows-msvc \
// RUN:        -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR-PER-TARGET %s
// PRINT-RUNTIME-DIR-PER-TARGET: lib{{/|\\}}x86_64-pc-windows-msvc
