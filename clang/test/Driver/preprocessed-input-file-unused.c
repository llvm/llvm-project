// Regression test for a condition inversion bug that caused
// warn_drv_preprocessed_input_file_unused to never mention which option
// determined the final compilation phase.

// RUN: %clang -E %S/Inputs/preprocessed-input-file-unused.i -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-E %s
// CHECK-E: warning: {{.*}}preprocessed-input-file-unused.i: previously preprocessed input unused when 'E' is present [-Wunused-command-line-argument]

// RUN: %clang -M %S/Inputs/preprocessed-input-file-unused.i -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-M %s
// CHECK-M: warning: {{.*}}preprocessed-input-file-unused.i: previously preprocessed input unused when 'M' is present [-Wunused-command-line-argument]

// RUN: %clang -MM %S/Inputs/preprocessed-input-file-unused.i -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MM %s
// CHECK-MM: warning: {{.*}}preprocessed-input-file-unused.i: previously preprocessed input unused when 'MM' is present [-Wunused-command-line-argument]
