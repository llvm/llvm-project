// Test this without pch.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -verify -fsyntax-only \
// RUN:                                           -std=c23 -include %S/Inputs/tmo_allocation_late_attr.h %s
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -verify -emit-llvm -o - \
// RUN:                                           -std=c23 -include %S/Inputs/tmo_allocation_late_attr.h %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation_late_attr.h
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include-pch %t -emit-llvm -o - %s | FileCheck %s

// Test with pch and remarks.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation_late_attr.h
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include-pch %t -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -verify -fsyntax-only \
// RUN:                                                         -std=c23 -include-pch %t %s

static void call_in_pch_function1(void) {
    in_pch_function1();
}
// CHECK-LABEL: @in_pch_function1()
// CHECK-LABEL: call ptr @malloc(i64 noundef 4)

static void call_in_pch_function2(void) {
    in_pch_function2();
}
// CHECK-LABEL: @in_pch_function2()
// CHECK: call ptr @typed_malloc(i64 noundef 8, i64 noundef [[GENERICDATA32_DESC:72058145178419728]])

void out_of_pch_function() {
  int *iptr1 = malloc(sizeof(int) * 3); // #iptr1
  // expected-remark@#iptr1 {{passing TMO information for array of type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#iptr1 {{inferred array of 'int' from expression 'sizeof(int) * 3'}}
  // expected-note@#iptr1 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
}
// CHECK-LABEL: @out_of_pch_function()
// CHECK: call ptr @typed_malloc(i64 noundef 12, i64 noundef [[GENERICDATA32_DESC]])
// CHECK: !{!"type-descriptor", !"[[GENERICDATA32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
