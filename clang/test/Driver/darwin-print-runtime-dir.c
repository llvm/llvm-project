// Regression test. Previously the output returned the full OS name
// (e.g. `darwin20.3.0`) instead of just `darwin`.

// RUN: %clang -print-runtime-dir --target=x86_64-apple-darwin20.3.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=x86_64-apple-macosx11.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=arm64-apple-ios14.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=arm64-apple-tvos14.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-runtime-dir --target=arm64-apple-watchos5.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// PRINT-RUNTIME-DIR: lib{{/|\\}}darwin{{$}}
