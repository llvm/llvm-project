// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wincompatible-library-redeclaration -fclangir -fclangir-idiom-recognizer="remarks=found-calls" -emit-cir -clangir-verify-diagnostics %s -o %t.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wincompatible-library-redeclaration -fclangir -fclangir-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-before=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wincompatible-library-redeclaration -fclangir -fclangir-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=RAISED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wincompatible-library-redeclaration -fclangir -fclangir-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=LOWERED

#include "Inputs/string.h"

// === CALL RECOGNITION, RAISING AND LOWERING PREPARE ===
int test_strlen(const char *s) {
    // CIR:    {{.*}} cir.call @strlen(
    // RAISED: {{.*}} cir.std.strlen(
    // LOWERED: {{.*}} cir.call @strlen(
    return strlen(s); // expected-remark {{found call to strlen()}}
}
