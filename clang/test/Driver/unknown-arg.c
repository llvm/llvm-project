// RUN: not %clang %s -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option 2>&1 | \
// RUN: FileCheck %s

// CHECK: unknown argument: '-cake-is-lie'
// CHECK: unknown argument: '-%0'
// CHECK: unknown argument: '-%d'
// CHECK: unknown argument: '-HHHH'
// CHECK: unknown argument: '-munknown-to-clang-option'
// CHECK: unknown argument: '-print-stats'
// CHECK: unknown argument: '-funknown-to-clang-option'


// RUN: %clang -S %s -o %t.s  -Wunknown-to-clang-option 2>&1 | FileCheck --check-prefix=IGNORED %s

// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
