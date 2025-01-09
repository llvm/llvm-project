// RUN: clang-tblgen --gen-clang-diags-enums -I%S %s 2>&1 | FileCheck %s
include "DiagnosticBase.inc"

def DupeNames1 : Error<"%enum_select<DupeName>{}0">;
def DupeNames2 : Error<"%enum_select<DupeName>{}0">;
// CHECK: error: Duplicate enumeration name 'DupeName'
// CHECK-NEXT: def DupeNames2
// CHECK: note: Previous diagnostic is here
// CHECK-NEXT: def DupeNames1

def DupeValue : Error<"%enum_select<DupeValue>{%Name{V1}|%Name{V2}}0">;
// CHECK: error: Duplicate enumerator name 'Name'

def EnumValNotExpected : Error<"%enum_select{V1|%Val2{V2}}0">;
// CHECK: expected '<' after enum_select

