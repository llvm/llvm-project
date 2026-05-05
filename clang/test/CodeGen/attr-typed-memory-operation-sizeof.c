// RUN: %clang_cc1 -Rtmo-remarks -verify=tmo,tmowarn  -Wtyped-memory-inference-failure -fsyntax-only \
// RUN:               -ftyped-memory-operations -DTMO=1 -triple x86_64-apple-macos -nostdsysteminc -O0 -disable-llvm-passes %s
// RUN: %clang_cc1 -Rtmo-remarks -verify=notmoremarks -Wtyped-memory-inference-failure -fsyntax-only \
// RUN:            -fno-typed-memory-operations -DTMO=0 -triple x86_64-apple-macos -nostdsysteminc -O0 -disable-llvm-passes %s
// RUN: %clang_cc1    -ftyped-memory-operations -DTMO=1 -triple x86_64-apple-macos -nostdsysteminc -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK          %s
// RUN: %clang_cc1 -fno-typed-memory-operations -DTMO=0 -triple x86_64-apple-macos -nostdsysteminc -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DISABLED %s
// notmoremarks-no-diagnostics
#if TMO
_Static_assert(__has_feature(typed_memory_operations), "");
#else
_Static_assert(!__has_feature(typed_memory_operations), "");
#endif

#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))
typedef unsigned int uint32_t;

void *typed_real_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size) _TYPED(typed_real_malloc, 1);

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *my_malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);
void *f() {
   return my_malloc(sizeof(__UINT32_TYPE__) * 2 * 2); // #alloc1
  // tmo-remark@#alloc1 {{passing TMO information for array of type 'unsigned int' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc1 {{inferred array of 'unsigned int' from expression 'sizeof(unsigned int) * 2 * 2'}}
  // tmo-note@#alloc1 {{encoding array of 'unsigned int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: call ptr @typed_malloc(i64 noundef 16, i64 noundef [[ARRAY32_DESC:72058145178419728]])
}
void *g() {
   return my_malloc(sizeof(__UINT32_TYPE__) * 2); // #alloc2
  // tmo-remark@#alloc2 {{passing TMO information for array of type 'unsigned int' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc2 {{inferred array of 'unsigned int' from expression 'sizeof(unsigned int) * 2'}}
  // tmo-note@#alloc2 {{encoding array of 'unsigned int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: call ptr @typed_malloc(i64 noundef 8, i64 noundef [[ARRAY32_DESC]])
}
void *h() {
   return my_malloc(sizeof(__UINT32_TYPE__) * 4 * 2); // #alloc3
  // tmo-remark@#alloc3 {{passing TMO information for array of type 'unsigned int' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc3 {{inferred array of 'unsigned int' from expression 'sizeof(unsigned int) * 4 * 2'}}
  // tmo-note@#alloc3 {{encoding array of 'unsigned int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: call ptr @typed_malloc(i64 noundef 32, i64 noundef [[ARRAY32_DESC]])
}

void test_expression_sizeof_uint() {
  malloc(sizeof(uint32_t)); // #alloc4
  // tmo-remark@#alloc4 {{passing TMO information for type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc4 {{inferred 'uint32_t' (aka 'unsigned int') from expression 'sizeof(uint32_t)'}}
  // tmo-note@#alloc4 {{encoding 'uint32_t' (aka 'unsigned int') as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC:72057870300512784]])
  // CHECK-DISABLED: %call = call ptr @malloc
  malloc(sizeof(uint32_t) * 2); // #alloc5
  // tmo-remark@#alloc5 {{passing TMO information for array of type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc5 {{inferred array of 'uint32_t' (aka 'unsigned int') from expression 'sizeof(uint32_t) * 2'}}
  // tmo-note@#alloc5 {{encoding array of 'uint32_t' (aka 'unsigned int') as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[ARRAY32_DESC]])
  // CHECK-DISABLED: %call1 = call ptr @malloc
  malloc(sizeof(uint32_t) * 2 * 2); // #alloc6
  // tmo-remark@#alloc6 {{passing TMO information for array of type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc6 {{inferred array of 'uint32_t' (aka 'unsigned int') from expression 'sizeof(uint32_t) * 2 * 2'}}
  // tmo-note@#alloc6 {{encoding array of 'uint32_t' (aka 'unsigned int') as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 16, i64 noundef [[ARRAY32_DESC]])
  // CHECK-DISABLED: %call2 = call ptr @malloc
  malloc(sizeof(uint32_t) * 4 * 2); // #alloc7
  // tmo-remark@#alloc7 {{passing TMO information for array of type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc7 {{inferred array of 'uint32_t' (aka 'unsigned int') from expression 'sizeof(uint32_t) * 4 * 2'}}
  // tmo-note@#alloc7 {{encoding array of 'uint32_t' (aka 'unsigned int') as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call3 = call ptr @typed_real_malloc(i64 noundef 32, i64 noundef [[ARRAY32_DESC]])
  // CHECK-DISABLED: %call3 = call ptr @malloc
}

void *typed_real_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size, unsigned long long);
void *calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size) _TYPED(typed_real_calloc, 2);

// CHECK: test_expression_sizeof_calloc
void test_expression_sizeof_calloc() {
  calloc(5, sizeof(int)); // #alloc8
  // tmo-remark@#alloc8 {{passing TMO information for type 'int' to 'typed_real_calloc' (retargeted from 'calloc')}}
  // tmo-note@#alloc8 {{inferred 'int' from expression 'sizeof(int)'}}
  // tmo-note@#alloc8 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_real_calloc(i64 noundef 5, i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call = call ptr @calloc
  calloc(1, sizeof(unsigned) * 4 * 2); // #alloc9
  // tmo-remark@#alloc9 {{passing TMO information for array of type 'unsigned int' to 'typed_real_calloc' (retargeted from 'calloc')}}
  // tmo-note@#alloc9 {{inferred array of 'unsigned int' from expression 'sizeof(unsigned int) * 4 * 2'}}
  // tmo-note@#alloc9 {{encoding array of 'unsigned int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_real_calloc(i64 noundef 1, i64 noundef 32, i64 noundef [[ARRAY32_DESC]])
  // CHECK-DISABLED: %call1 = call ptr @calloc
}
// CHECK: test_expression_sizeof_uint_arbitrary_expr
void test_expression_sizeof_uint_arbitrary_expr() {
  malloc(sizeof(uint32_t) + 0); // #alloc10
  // tmo-remark@#alloc10 {{passing TMO information for type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc10 {{inferred indeterminate set of {'uint32_t' (aka 'unsigned int')} from expression 'sizeof(uint32_t) + 0'}}
  // tmo-note@#alloc10 {{encoding 'uint32_t' (aka 'unsigned int') as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef 72057595422605840)
  // CHECK-DISABLED: %call = call ptr @malloc
  malloc(sizeof(uint32_t) + sizeof(uint32_t)); // #alloc11
  // tmo-remark@#alloc11 {{passing TMO information for type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc11 {{inferred tuple of ('uint32_t' (aka 'unsigned int'), 'uint32_t' (aka 'unsigned int')) from expression 'sizeof(uint32_t) + sizeof(uint32_t)'}}
  // tmo-note@#alloc11 {{encoding 'uint32_t' (aka 'unsigned int') as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call1 = call ptr @malloc
  malloc(sizeof(uint32_t) * (2 + 2)); // #alloc12
  // tmo-remark@#alloc12 {{passing TMO information for array of type 'uint32_t' (aka 'unsigned int') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc12 {{inferred array of 'uint32_t' (aka 'unsigned int') from expression 'sizeof(uint32_t) * (2 + 2)'}}
  // tmo-note@#alloc12 {{encoding array of 'uint32_t' (aka 'unsigned int') as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 16, i64 noundef [[ARRAY32_DESC]])
  // CHECK-DISABLED: %call2 = call ptr @malloc
}

struct T {
  int i;
  void (*f)();
  void *p;
};
// CHECK: test_struct_t
void test_struct_t() {
  malloc(sizeof(struct T) * 2 * 2); // #alloc13
  // tmo-remark@#alloc13 {{passing TMO information for array of type 'struct T' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc13 {{inferred array of 'struct T' from expression 'sizeof(struct T) * 2 * 2'}}
  // tmo-note@#alloc13 {{encoding array of 'struct T' as 74309946902329154. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 3294902082 }}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 96, i64 noundef [[ARRAY_T_DESC:74309946902329154]])
  // CHECK-DISABLED: %call = call ptr @malloc
  malloc(sizeof(struct T) * 4); // #alloc14
  // tmo-remark@#alloc14 {{passing TMO information for array of type 'struct T' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc14 {{inferred array of 'struct T' from expression 'sizeof(struct T) * 4'}}
  // tmo-note@#alloc14 {{encoding array of 'struct T' as 74309946902329154. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 3294902082 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 96, i64 noundef [[ARRAY_T_DESC]])
  // CHECK-DISABLED: %call1 = call ptr @malloc
  malloc(sizeof(struct T) + sizeof(struct T) + sizeof(struct T) + sizeof(struct T)); // #alloc15
  // tmo-remark@#alloc15 {{passing TMO information for type 'struct T' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc15 {{inferred tuple of ('struct T', 'struct T', 'struct T', 'struct T') from expression 'sizeof(struct T) + sizeof(struct T) + sizeof(struct T) + sizeof(struct T)'}}
  // tmo-note@#alloc15 {{encoding 'struct T' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 96, i64 noundef [[T_DESC:74309672024422210]])
  // CHECK-DISABLED: %call2 = call ptr @malloc
}

// CHECK: !{!"type-descriptor", !"[[ARRAY32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_T_DESC]]", !"3294902082", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[T_DESC]]", !"3294902082", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
