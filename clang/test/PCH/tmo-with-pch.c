// Test this without pch.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include %S/Inputs/tmo_allocation.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -Wtyped-memory-inference-failure  -verify -fsyntax-only \
// RUN:                                           -std=c23 -include %S/Inputs/tmo_allocation.h -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation.h
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include-pch %t -emit-llvm -o - %s | FileCheck %s

// Test with pch and remarks.
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation.h
// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include-pch %t -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -Wtyped-memory-inference-failure -verify -fsyntax-only \
// RUN:                                                         -std=c23 -include-pch %t %s

// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation.h
// RUN: not %clang_cc1 -x c -std=c23 -include-pch %t %s 2>&1 | FileCheck --check-prefix=PCH_HAS_TMO %s
// PCH_HAS_TMO: Typed Memory Operations Callsite Rewriting was enabled in precompiled file
// RUN: %clang_cc1 -x c -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation.h
// RUN: not %clang_cc1 -x c -ftyped-memory-operations -std=c23 -include-pch %t %s 2>&1 | FileCheck --check-prefix=PCH_NO_TMO %s
// PCH_NO_TMO: Typed Memory Operations Callsite Rewriting was disabled in precompiled file
static void call_in_pch_function(void) {
    in_pch_function();
}
// CHECK-LABEL: in_pch_function
// CHECK: call ptr @typed_malloc(i64 noundef 1000, i64 noundef [[LOC1_DESC:[0-9]+]])
// CHECK: call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC:72057870300512784]])
// CHECK: call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
// CHECK: call ptr @typed_malloc(i64 noundef 100, i64 noundef [[GENERICDATA32_DESC]])
// CHECK: call ptr @typed_malloc(i64 noundef 24, i64 noundef [[S1_DESC:74309672738655766]])
// CHECK: call ptr @typed_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
// CHECK: call ptr @typed_malloc(i64 noundef 100, i64 noundef [[S1_DESC]])

static void call_in_pch_fam_function(__SIZE_TYPE__ n) {
    in_pch_fam_function(n);
}
// CHECK-LABEL: in_pch_fam_function
// CHECK: call ptr @typed_malloc({{.*}}, i64 noundef [[ELEM_ARRAY_DESC:72058145178419728]])
// CHECK: call ptr @typed_malloc({{.*}}, i64 noundef [[PREFIX_HEADER_HPA_DESC:72058694934233616]])
// CHECK: call ptr @typed_malloc({{.*}}, i64 noundef [[PREFIX_HEADER_HPA_DESC]])
// CHECK: call ptr @typed_malloc(i64 noundef 28, i64 noundef [[S1_DESC]])
// CHECK: call ptr @typed_malloc({{.*}}, i64 noundef [[ELEM_INDETERMINATE_DESC:72057595422605840]])

// CHECK-LABEL: out_of_pch_function
void out_of_pch_function() {
  __SIZE_TYPE__ n;
  int *iptr1 = malloc(sizeof(int)); // #iptr1
  // CHECK: call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC:72057870300512784]])
  // expected-remark@#iptr1 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#iptr1 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#iptr1 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  int *iptr2 = (int *)malloc(sizeof(int)); // #iptr2
  // CHECK: call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // expected-remark@#iptr2 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#iptr2 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#iptr2 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  int *iptr3 = (int *)malloc(10); // #iptr3
  // CHECK: call ptr @typed_malloc(i64 noundef 10, i64 noundef [[GENERICDATA32_DESC]])
  // expected-remark@#iptr3 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#iptr3 {{inferred 'int' from cast of result from call to '(int *)malloc(10)'}}
  // expected-note@#iptr3 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  struct S1 *s1ptr1 = malloc(sizeof(struct S1)); // #s1ptr1
  // CHECK: call ptr @typed_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  // expected-remark@#s1ptr1 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#s1ptr1 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#s1ptr1 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1 *s1ptr2 = (struct S1 *)malloc(sizeof(struct S1)); // #s1ptr2
  // CHECK: call ptr @typed_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  // expected-remark@#s1ptr2 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#s1ptr2 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#s1ptr2 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1 *s1ptr3 = (struct S1 *)malloc(100); // #s1ptr3
  // CHECK: call ptr @typed_malloc(i64 noundef 100, i64 noundef [[S1_DESC]])
  // expected-remark@#s1ptr3 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#s1ptr3 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)malloc(100)'}}
  // expected-note@#s1ptr3 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  void *tuple1 = malloc(sizeof(struct S1) + sizeof(struct Elem)); // #tuple1
  // CHECK: call ptr @typed_malloc(i64 noundef 28, i64 noundef [[S1_DESC]])
  // expected-remark@#tuple1 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#tuple1 {{inferred tuple of ('struct S1', 'struct Elem') from expression 'sizeof(struct S1) + sizeof(struct Elem)'}}
  // expected-note@#tuple1 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  void *unknown1 = malloc(sizeof(struct Elem) + n); // #unknown1
  // CHECK: call ptr @typed_malloc({{.*}}, i64 noundef [[ELEM_INDETERMINATE_DESC]])
  // expected-remark@#unknown1 {{passing TMO information for type 'struct Elem' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#unknown1 {{inferred indeterminate set of {'struct Elem'} from expression 'sizeof(struct Elem) + n'}}
  // expected-note@#unknown1 {{encoding 'struct Elem' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
}

// CHECK: !{!"type-descriptor", !"[[LOC1_DESC]]", !"[[LOC1_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ELEM_ARRAY_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[PREFIX_HEADER_HPA_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22HeaderPrefixedArray\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ELEM_INDETERMINATE_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
