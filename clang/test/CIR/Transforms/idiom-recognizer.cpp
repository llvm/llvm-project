// RUN: %clang_cc1 -fclangir -emit-cir -maiir --aiir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR:  IR Dump After IdiomRecognizer (cir-idiom-recognizer)
