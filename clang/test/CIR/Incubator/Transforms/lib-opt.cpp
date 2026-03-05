// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-idiom-recognizer -fclangir-lib-opt -emit-cir -mmlir --mlir-print-ir-after-all %s -o - 2>&1 | FileCheck %s -check-prefix=CIR

// CIR: IR Dump After LibOpt (cir-lib-opt)