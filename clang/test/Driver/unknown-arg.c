// RUN: %clang %s -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -### 2>&1 | \
// RUN: FileCheck %s
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -### -c -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Werror=unknown-argument -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL-ERROR
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Wno-unknown-argument -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=SILENT

// CHECK: error: unknown argument: '-cake-is-lie'
// CHECK: error: unknown argument: '-%0'
// CHECK: error: unknown argument: '-%d'
// CHECK: error: unknown argument: '-HHHH'
// CHECK: error: unknown argument: '-munknown-to-clang-option'
// CHECK: error: unknown argument: '-print-stats'
// CHECK: error: unknown argument: '-funknown-to-clang-option'
// CL: warning: unknown argument ignored in clang-cl: '-cake-is-lie'
// CL: warning: unknown argument ignored in clang-cl: '-%0'
// CL: warning: unknown argument ignored in clang-cl: '-%d'
// CL: warning: unknown argument ignored in clang-cl: '-HHHH'
// CL: warning: unknown argument ignored in clang-cl: '-munknown-to-clang-option'
// CL: warning: unknown argument ignored in clang-cl: '-print-stats'
// CL: warning: unknown argument ignored in clang-cl: '-funknown-to-clang-option'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-cake-is-lie'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-%0'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-%d'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-HHHH'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-munknown-to-clang-option'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-print-stats'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-funknown-to-clang-option'
// SILENT-NOT: error:
// SILENT-NOT: warning:


// RUN: %clang -S %s -o %t.s  -Wunknown-to-clang-option 2>&1 | FileCheck --check-prefix=IGNORED %s

// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
