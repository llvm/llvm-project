// Invalid section specifiers passed via -fclang-bss-section= and
// -fclang-data-section= are diagnosed. The MachO section specifier parser is
// the only one that rejects names, so use a Darwin target.

// RUN: not %clang_cc1 -fsyntax-only -triple arm64-apple-ios \
// RUN:   -fclang-bss-section=badname %s 2>&1 | FileCheck %s --check-prefix=BSS
// RUN: not %clang_cc1 -fsyntax-only -triple arm64-apple-ios \
// RUN:   -fclang-data-section=badname %s 2>&1 | FileCheck %s --check-prefix=DATA

// Valid MachO section specifiers are accepted.
// RUN: %clang_cc1 -fsyntax-only -verify -triple arm64-apple-ios \
// RUN:   -fclang-bss-section=__DATA,__my_bss \
// RUN:   -fclang-data-section=__DATA,__my_data %s

// BSS: error: argument to '-fclang-bss-section=' is not valid for this target: mach-o section specifier requires a segment and section separated by a comma
// DATA: error: argument to '-fclang-data-section=' is not valid for this target: mach-o section specifier requires a segment and section separated by a comma

// expected-no-diagnostics
int zeroed;
int initialized = 1;