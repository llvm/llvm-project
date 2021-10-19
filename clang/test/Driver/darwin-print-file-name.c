// Regression test. Previously Clang just returned the library name instead
// of the full path.

// RUN: %clang -print-file-name=libclang_rt.osx.a --target=x86_64-apple-darwin20.3.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// RUN: %clang -print-file-name=libclang_rt.osx.a --target=x86_64-apple-macosx11.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR %s

// PRINT-RUNTIME-DIR: lib{{/|\\}}darwin{{/|\\}}libclang_rt.osx.a

// RUN: %clang -print-file-name=libclang_rt.ios.a --target=arm64-apple-ios14.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR-IOS %s
// PRINT-RUNTIME-DIR-IOS: lib{{/|\\}}darwin{{/|\\}}libclang_rt.ios.a

// RUN: %clang -print-file-name=libclang_rt.tvos.a --target=arm64-apple-tvos14.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR-TVOS %s
// PRINT-RUNTIME-DIR-TVOS: lib{{/|\\}}darwin{{/|\\}}libclang_rt.tvos.a

// RUN: %clang -print-file-name=libclang_rt.watchos.a --target=arm64-apple-watchos5.0.0 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-RUNTIME-DIR-WATCHOS %s
// PRINT-RUNTIME-DIR-WATCHOS: lib{{/|\\}}darwin{{/|\\}}libclang_rt.watchos.a
