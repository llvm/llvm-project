// RUN: %clang_cc1 -fclangir -O1 -emit-cir -mmlir --mlir-print-ir-after-all -clangir-lib-opt=all %s -o - 2>&1 | FileCheck %s -check-prefix=ENABLED
// RUN: %clang_cc1 -fclangir -O1 -emit-cir -mmlir --mlir-print-ir-after-all %s -o - 2>&1 | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fclangir -O0 -emit-cir -mmlir --mlir-print-ir-after-all -clangir-lib-opt=all %s -o - 2>&1 | FileCheck %s -check-prefix=DISABLED

// ENABLED: IR Dump After LibOpt: cir-lib-opt
// DISABLED-NOT: IR Dump After LibOpt: cir-lib-opt
