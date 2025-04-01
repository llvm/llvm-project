// RUN: not clang-tblgen --gen-clang-diags-enums -DERROR1 -I%S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C1
// RUN: not clang-tblgen --gen-clang-diags-enums -DERROR2 -I%S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C2
// RUN: not clang-tblgen --gen-clang-diags-enums -DERROR3 -I%S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C3
// RUN: not clang-tblgen --gen-clang-diags-enums -DERROR4 -I%S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C4
include "DiagnosticBase.inc"

// No real reason to diagnose these, the namespace generated as the
// 'enumeration' name will never conflict with the enumerator.
def EnumerationEnumeratorDupe : Error<"%enum_select<Matchy>{%Matchy{haha}}0">;

// Enumerator values aren't required, though this does seem kind of silly/not
// particularly useful?
def NoEnumerators : Error<"%enum_select<Name>{foo|bar|baz}0">;

def DupeNames1 : Error<"%enum_select<DupeName>{}0">;
def DupeNames2 : Error<"%enum_select<DupeName>{}0">;
// CHECK: error: Duplicate enumeration name 'DupeName'
// CHECK-NEXT: def DupeNames2
// CHECK: note: Previous diagnostic is here
// CHECK-NEXT: def DupeNames1

def DupeValue : Error<"%enum_select<DupeValue>{%DName{V1}|%DName{V2}}0">;
// CHECK: error: Duplicate enumerator name 'DName'

#ifdef ERROR1
def EnumValNotExpected : Error<"%enum_select{V1|%Val2{V2}}0">;
// C1: expected '<' after enum_select
#endif

#ifdef ERROR2
def SelectWithArrow : Error<"%select<Something>{V1|%Val2{V2}}0">;
// C2: modifier '<' syntax not valid with %select
#endif

#ifdef ERROR3
// Missing closing > after the name of the enumeration
def MissingClosing : Error<"%enum_select<MissingClosingName{}0">;
// C3: expected '>' while parsing %enum_select
#endif

#ifdef ERROR4
// Missing { after the name of an enumerator
def MissingTextAfterEnumerator: Error<"%enum_select<Name>{%OtherName|foo}0">;
// C4: expected '{' while parsing %enum_select
#endif
