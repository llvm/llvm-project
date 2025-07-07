// RUN: clang-tblgen --gen-clang-diags-enums -I%S %s 2>&1 | FileCheck %s
include "DiagnosticBase.inc"

def Diag : Error<"%enum_select<EnumName>{%Val1{V1}|%Val2{V2}|%Val3{V3}}0">;
// CHECK: DIAG_ENUM(EnumName)
// CHECK-NEXT: DIAG_ENUM_ITEM(0, Val1)
// CHECK-NEXT: DIAG_ENUM_ITEM(1, Val2)
// CHECK-NEXT: DIAG_ENUM_ITEM(2, Val3)
// CHECK-NEXT: DIAG_ENUM_END()

// These are OK, we permit missing values since they might not be useful.
def Missing1 : Error<"%enum_select<DupeEnumName1>{V1|%Val2{V2}|%Val3{V3}}0">;
// CHECK: DIAG_ENUM(DupeEnumName1)
// CHECK-NEXT: DIAG_ENUM_ITEM(1, Val2)
// CHECK-NEXT: DIAG_ENUM_ITEM(2, Val3)
// CHECK-NEXT: DIAG_ENUM_END()
def Missing2 : Error<"%enum_select<DupeEnumName2>{%Val1{V1}|V2|%Val3{V3}}0">;
// CHECK: DIAG_ENUM(DupeEnumName2)
// CHECK-NEXT: DIAG_ENUM_ITEM(0, Val1)
// CHECK-NEXT: DIAG_ENUM_ITEM(2, Val3)
// CHECK-NEXT: DIAG_ENUM_END()
def Missing3 : Error<"%enum_select<DupeEnumName3>{%Val1{V1}|%Val2{V2}|V3}0">;
// CHECK: DIAG_ENUM(DupeEnumName3)
// CHECK-NEXT: DIAG_ENUM_ITEM(0, Val1)
// CHECK-NEXT: DIAG_ENUM_ITEM(1, Val2)
// CHECK-NEXT: DIAG_ENUM_END()
