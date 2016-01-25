// RUN: not %clang %s -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option 2>&1 | \
// RUN: FileCheck %s
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL
// RUN: not %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Werror=unknown-argument -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Wno-unknown-argument -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=SILENT --allow-empty

// CHECK: unknown argument: '-cake-is-lie'
// CHECK: unknown argument: '-%0'
// CHECK: unknown argument: '-%d'
// CHECK: unknown argument: '-HHHH'
// CHECK: unknown argument: '-munknown-to-clang-option'
// CHECK: unknown argument: '-print-stats'
// CHECK: unknown argument: '-funknown-to-clang-option'
// CL: unknown argument ignored in clang-cl: '-cake-is-lie'
// CL: unknown argument ignored in clang-cl: '-%0'
// CL: unknown argument ignored in clang-cl: '-%d'
// CL: unknown argument ignored in clang-cl: '-HHHH'
// CL: unknown argument ignored in clang-cl: '-munknown-to-clang-option'
// CL: unknown argument ignored in clang-cl: '-print-stats'
// CL: unknown argument ignored in clang-cl: '-funknown-to-clang-option'
// SILENT-NOT: error
// SILENT-NOT: warning


// RUN: %clang -S %s -o %t.s  -Wunknown-to-clang-option 2>&1 | FileCheck --check-prefix=IGNORED %s

// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
