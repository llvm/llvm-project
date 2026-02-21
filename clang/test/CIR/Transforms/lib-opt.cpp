// RUN: %clang_cc1 -fclangir -O1 -emit-cir -mmlir --mlir-print-ir-after-all -fclangir-lib-opt %s -o - 2>&1 | FileCheck %s -check-prefix=CIR

// CIR: IR Dump After LibOpt (cir-lib-opt)
