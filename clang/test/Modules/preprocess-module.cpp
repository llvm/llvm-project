// RUN: rm -rf %t

// RUN: not %clang_cc1 -fmodules -fmodule-name=file -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E 2>&1 | FileCheck %s --check-prefix=MISSING-FWD
// MISSING-FWD: module 'fwd' is needed

// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodules-cache-path=%t -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E | FileCheck %s
// CHECK: # 1 "<module-includes>"
// CHECK: # 1 "{{.*}}file.h" 1
// CHECK: struct __FILE;
// CHECK: #include "fwd.h" /* clang -E: implicit import for module fwd */
// CHECK: typedef struct __FILE FILE;
// CHECK: # 2 "<module-includes>" 2
