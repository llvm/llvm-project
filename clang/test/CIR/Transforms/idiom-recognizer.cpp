// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR

// CIR:  IR Dump After IdiomRecognizer (cir-idiom-recognizer)

