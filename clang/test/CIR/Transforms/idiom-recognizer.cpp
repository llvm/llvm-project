// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=PASS_ENABLED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -fclangir-idiom-recognizer="remarks=found-calls" -clangir-verify-diagnostics %s -o %t2.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-before=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=BEFORE-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-LOWERING-PREPARE

// PASS_ENABLED:  IR Dump After IdiomRecognizer (cir-idiom-recognizer)

#include "std-cxx.h"

int test_find(unsigned char n = 3)
{
    unsigned num_found = 0;
    std::array<unsigned char, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto f = std::find(v.begin(), v.end(), n); // expected-remark {{found call to std::find()}}
    // BEFORE-IDIOM: {{.*}} cir.call @_ZSt4findINSt5arrayIhLj9EE8iteratorEhET_S3_S3_RKT0_(
    // AFTER-IDIOM: {{.*}} cir.std.find(@_ZSt4findINSt5arrayIhLj9EE8iteratorEhET_S3_S3_RKT0_,
    // AFTER-LOWERING-PREPARE: {{.*}} cir.call @_ZSt4findINSt5arrayIhLj9EE8iteratorEhET_S3_S3_RKT0_(

    if (f != v.end())
        num_found++;
    return num_found;
}