// RUN: %clang_cc1 -ftyped-memory-operations -triple x86_64-apple-macos -nostdsysteminc -O0 %s \
// RUN:            -Rtmo-remarks -verify -fsyntax-only
// RUN: %clang_cc1 -ftyped-memory-operations -triple x86_64-apple-macos -nostdsysteminc -O0 %s \
// RUN:            -disable-llvm-passes -emit-llvm -o - | FileCheck --check-prefix=CHECK %s

#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_real_malloc(__SIZE_TYPE__ size, unsigned long long);

struct S {
  int i;
  void (*f)(int);
  void *ptr;
};

void *malloc(__SIZE_TYPE__ size);
void test_malloc1() {
  // CHECK-LABEL: void @test_malloc1()
  struct S* s1 = malloc(sizeof(struct S));
  // CHECK: @malloc(i64 noundef 24)
  struct S* s2 = malloc(sizeof(struct S) * 5);
  // CHECK: @malloc(i64 noundef 120)
  struct S* s3 = (struct S*)malloc(100);
  // CHECK: @malloc(i64 noundef 100)
  void* s4 = malloc(sizeof(struct S));
  // CHECK: @malloc(i64 noundef 24)
  void* s5 = (struct S*)malloc(100);
  // CHECK: @malloc(i64 noundef 100)
}

void *malloc(__SIZE_TYPE__ size) _TYPED(typed_real_malloc, 1);

void test_malloc2() {
  // CHECK-LABEL: @test_malloc2()
  struct S* s1 = malloc(sizeof(struct S)); //#test2_s1
  // CHECK: @typed_real_malloc(i64 noundef 24, i64 noundef [[S_DESC:74309672024422210]])
  // expected-remark@#test2_s1 {{passing TMO information for type 'struct S' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#test2_s1 {{inferred 'struct S' from expression 'sizeof(struct S)'}}
  // expected-note@#test2_s1 {{encoding 'struct S' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
  struct S* s2 = malloc(sizeof(struct S) * 5); //#test2_s2
  // CHECK: @typed_real_malloc(i64 noundef 120, i64 noundef [[S_DESC_ARRAY:74309946902329154]])
  // expected-remark@#test2_s2 {{passing TMO information for array of type 'struct S' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#test2_s2 {{inferred array of 'struct S' from expression 'sizeof(struct S) * 5'}}
  // expected-note@#test2_s2 {{encoding array of 'struct S' as 74309946902329154. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 3294902082 }}}
  struct S* s3 = (struct S*)malloc(100); //#test2_s3
  // CHECK: @typed_real_malloc(i64 noundef 100, i64 noundef [[S_DESC]])
  // expected-remark@#test2_s3 {{passing TMO information for type 'struct S' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#test2_s3 {{inferred 'struct S' from cast of result from call to '(struct S *)malloc(100)'}}
  // expected-note@#test2_s3 {{encoding 'struct S' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
  void* s4 = malloc(sizeof(struct S)); //#test2_s4
  // CHECK: @typed_real_malloc(i64 noundef 24, i64 noundef [[S_DESC]])
  // expected-remark@#test2_s4 {{passing TMO information for type 'struct S' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#test2_s4 {{inferred 'struct S' from expression 'sizeof(struct S)'}}
  // expected-note@#test2_s4 {{encoding 'struct S' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
  void* s5 = (struct S*)malloc(100); //#test2_s5
  // CHECK: @typed_real_malloc(i64 noundef 100, i64 noundef [[S_DESC]])
  // expected-remark@#test2_s5 {{passing TMO information for type 'struct S' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#test2_s5 {{inferred 'struct S' from cast of result from call to '(struct S *)malloc(100)'}}
  // expected-note@#test2_s5 {{encoding 'struct S' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
}

// CHECK: !{!"type-descriptor", !"[[S_DESC]]", !"3294902082", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S_DESC_ARRAY]]", !"3294902082", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
