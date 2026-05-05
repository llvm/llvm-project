// RUN: %clang_cc1 -Rtmo-remarks -verify -fsyntax-only \
// RUN:               -ftyped-memory-operations -DTMO=1 -DTMO_REMARKS -triple x86_64-apple-macos -nostdsysteminc -Wno-alloc-size -O0 %s
// RUN: %clang_cc1    -ftyped-memory-operations -DTMO=1 -triple x86_64-apple-macos -nostdsysteminc -Wno-alloc-size -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK          %s
// RUN: %clang_cc1 -fno-typed-memory-operations -DTMO=0 -triple x86_64-apple-macos -nostdsysteminc -Wno-alloc-size -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DISABLED %s
// RUN: %clang_cc1    -ftyped-memory-operations -DTMO=1 -triple x86_64-apple-macos -nostdsysteminc -Wno-alloc-size -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK          %s

#if TMO
_Static_assert(__has_feature(typed_memory_operations), "");
#else
_Static_assert(!__has_feature(typed_memory_operations), "");
#endif

#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *typed_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size, unsigned long long);
void typed_unusual_alloc(float, unsigned, unsigned long long, const char*);

void *my_malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);
void *my_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size) _TYPED(typed_calloc, 2);
void unusual_alloc(float, unsigned, const char*) _TYPED(typed_unusual_alloc, 2);

void *typed_real_malloc(__SIZE_TYPE__ size, unsigned long long);
void *typed_real_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size, unsigned long long);
void *typed_real_realloc(void *, __SIZE_TYPE__ size, unsigned long long);
void *typed_real_aligned_alloc(unsigned long align, unsigned long size, unsigned long long);
int typed_real_posix_memalign(void **, __SIZE_TYPE__, __SIZE_TYPE__, unsigned long long);

void *malloc(__SIZE_TYPE__ size) _TYPED(typed_real_malloc, 1);
void *calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size) _TYPED(typed_real_calloc, 2);
void *realloc(void *, __SIZE_TYPE__ size) _TYPED(typed_real_realloc, 2);
void *aligned_alloc(unsigned long align, unsigned long size) _TYPED(typed_real_aligned_alloc, 2);
int posix_memalign(void **, __SIZE_TYPE__, __SIZE_TYPE__) _TYPED(typed_real_posix_memalign, 3);

struct S1 {
  void *p;
  int i;
  int j;
  void (*fptr)();
};

struct S2 {
  int i;
};

union U1 {
  int i;
};

union U2 {
  int i;
  void* v;
};


// We use these functions to test order of evaluation in some of
// these tests as there the override forwarding is some logical
// argument reordering but we need to maintain the C semantic
// ordering.
extern int f1(void);
extern int f2(void);
typedef struct S1 S1Mk2;
void f(void);
void f(void) {
  // Basic codegen tests
  malloc(5); // #alloc1
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 5, i64 noundef [[LOC1_DESC:[0-9]+]])
  // CHECK-DISABLED: %call = call ptr @malloc(i64 noundef 5)
  malloc(sizeof(int)); // #alloc2
  // expected-remark@#alloc2 {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc2 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#alloc2 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC:72057870300512784]])
  // CHECK-DISABLED: %call1 = call ptr @malloc(i64 noundef 4)
  malloc(sizeof(struct S1)); // #alloc3
  // expected-remark@#alloc3 {{passing TMO information for type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc3 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#alloc3 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 24, i64 noundef [[S1_DESC:74309672738655766]])
  // CHECK-DISABLED: %call2 = call ptr @malloc(i64 noundef 24)
  malloc(sizeof(struct S1) * 5); // #alloc4
  // expected-remark@#alloc4 {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc4 {{inferred array of 'struct S1' from expression 'sizeof(struct S1) * 5'}}
  // expected-note@#alloc4 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call3 = call ptr @typed_real_malloc(i64 noundef 120, i64 noundef [[ARRAY_S1_DESC:74309947616562710]])
  // CHECK-DISABLED: %call3 = call ptr @malloc(i64 noundef 120)
  malloc(sizeof(struct S1) + sizeof(struct S2)); // #alloc5
  // expected-remark@#alloc5 {{passing TMO information for type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc5 {{inferred tuple of ('struct S1', 'struct S2') from expression 'sizeof(struct S1) + sizeof(struct S2)'}}
  // expected-note@#alloc5 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call4 = call ptr @typed_real_malloc(i64 noundef 28, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call4 = call ptr @malloc(i64 noundef 28)
  my_malloc(5); // #alloc6
  // CHECK: %call5 = call ptr @typed_malloc(i64 noundef 5, i64 noundef [[LOC2_DESC:[0-9]+]])
  // CHECK-DISABLED: %call5 = call ptr @my_malloc(i64 noundef 5)
  my_malloc(sizeof(int)); // #alloc7
  // expected-remark@#alloc7 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // expected-note@#alloc7 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#alloc7 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call6 = call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call6 = call ptr @my_malloc(i64 noundef 4)
  my_malloc(sizeof(S1Mk2)); // #alloc8
  // expected-remark@#alloc8 {{passing TMO information for type 'S1Mk2' (aka 'struct S1') to 'typed_malloc' (retargeted from 'my_malloc')}}
  // expected-note@#alloc8 {{inferred 'S1Mk2' (aka 'struct S1') from expression 'sizeof(S1Mk2)'}}
  // expected-note@#alloc8 {{encoding 'S1Mk2' (aka 'struct S1') as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call7 = call ptr @typed_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call7 = call ptr @my_malloc(i64 noundef 24)
  my_malloc(sizeof(struct S1) * 5); // #alloc9
  // expected-remark@#alloc9 {{passing TMO information for array of type 'struct S1' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // expected-note@#alloc9 {{inferred array of 'struct S1' from expression 'sizeof(struct S1) * 5'}}
  // expected-note@#alloc9 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call8 = call ptr @typed_malloc(i64 noundef 120, i64 noundef [[ARRAY_S1_DESC]])
  // CHECK-DISABLED: %call8 = call ptr @my_malloc(i64 noundef 120)
  my_malloc(sizeof(struct S1) + sizeof(union U1)); // #alloc10
  // expected-remark@#alloc10 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // expected-note@#alloc10 {{inferred tuple of ('struct S1', 'union U1') from expression 'sizeof(struct S1) + sizeof(union U1)'}}
  // expected-note@#alloc10 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call9 = call ptr @typed_malloc(i64 noundef 28, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call9 = call ptr @my_malloc(i64 noundef 28)

  // Ordering of argument evaluation
  my_calloc(f1(), f2()); // #alloc11
  // CHECK: %call10 = call i32 @f1()
  // CHECK: %conv = sext i32 %call10 to i64
  // CHECK: %call11 = call i32 @f2()
  // CHECK: %conv12 = sext i32 %call11 to i64
  // CHECK: %call13 = call ptr @typed_calloc(i64 noundef %conv, i64 noundef %conv12, i64 noundef [[LOC3_DESC:[0-9]+]])
  // CHECK-DISABLED: %call13 = call ptr @my_calloc
  my_calloc(f1(), sizeof(int) * f2()); // #alloc12
  // expected-remark@#alloc12 {{passing TMO information for array of type 'int' to 'typed_calloc' (retargeted from 'my_calloc')}}
  // expected-note@#alloc12 {{inferred array of 'int' from expression 'sizeof(int) * f2()'}}
  // expected-note@#alloc12 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call14 = call i32 @f1()
  // CHECK: %conv15 = sext i32 %call14 to i64
  // CHECK: %call16 = call i32 @f2()
  // CHECK: %conv17 = sext i32 %call16 to i64
  // CHECK: %mul = mul i64 4, %conv17
  // CHECK: %call18 = call ptr @typed_calloc(i64 noundef %conv15, i64 noundef %mul, i64 noundef [[ARRAY_INT32_DESC:72058145178419728]])
  // CHECK-DISABLED: %call18 = call ptr @my_calloc
  my_calloc(f1(), f2() * sizeof(double)); // #alloc13
  // expected-remark@#alloc13 {{passing TMO information for array of type 'double' to 'typed_calloc' (retargeted from 'my_calloc')}}
  // expected-note@#alloc13 {{inferred array of 'double' from expression 'f2() * sizeof(double)'}}
  // expected-note@#alloc13 {{encoding array of 'double' as 72058143796969239. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 3227415 }}}
  // CHECK: %call19 = call i32 @f1()
  // CHECK: %conv20 = sext i32 %call19 to i64
  // CHECK: %call21 = call i32 @f2()
  // CHECK: %conv22 = sext i32 %call21 to i64
  // CHECK: %mul23 = mul i64 %conv22, 8
  // CHECK: %call24 = call ptr @typed_calloc(i64 noundef %conv20, i64 noundef %mul23, i64 noundef [[ARRAY_DOUBLE_DESC:72058143796969239]])
  // CHECK-DISABLED: %call24 = call ptr @my_calloc

  // Order of arguments passed to target
  unusual_alloc(0.5, sizeof(int) * f1(), "womp"); // #alloc14
  // expected-remark@#alloc14 {{passing TMO information for array of type 'int' to 'typed_unusual_alloc' (retargeted from 'unusual_alloc')}}
  // expected-note@#alloc14 {{inferred array of 'int' from expression 'sizeof(int) * f1()'}}
  // expected-note@#alloc14 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call25 = call i32 @f1()
  // CHECK: %conv26 = sext i32 %call25 to i64
  // CHECK: %mul27 = mul i64 4, %conv26
  // CHECK: %conv28 = trunc i64 %mul27 to i32
  // CHECK: call void @typed_unusual_alloc(float noundef 5.000000e-01, i32 noundef %conv28, i64 noundef [[ARRAY_INT32_DESC]], ptr noundef @.str)
  // CHECK-DISABLED: call void @unusual_alloc

  malloc(sizeof(struct S1)); // #alloc15
  // expected-remark@#alloc15 {{passing TMO information for type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc15 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#alloc15 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call29 = call ptr @typed_real_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call29 = call ptr @malloc
  malloc(sizeof(struct S2)); // #alloc16
  // expected-remark@#alloc16 {{passing TMO information for type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc16 {{inferred 'struct S2' from expression 'sizeof(struct S2)'}}
  // expected-note@#alloc16 {{encoding 'struct S2' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call30 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call30 = call ptr @malloc
  malloc(sizeof(union U1)); // #alloc17
  // expected-remark@#alloc17 {{passing TMO information for type 'union U1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc17 {{inferred 'union U1' from expression 'sizeof(union U1)'}}
  // expected-note@#alloc17 {{encoding 'union U1' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call31 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call31 = call ptr @malloc
  malloc(sizeof(union U2)); // #alloc18
  // expected-remark@#alloc18 {{passing TMO information for type 'union U2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc18 {{inferred 'union U2' from expression 'sizeof(union U2)'}}
  // expected-note@#alloc18 {{encoding 'union U2' as 74344853696240142. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 594631182 }}}
  // CHECK: %call32 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[U2_DESC:74344853696240142]])
  // CHECK-DISABLED: %call32 = call ptr @malloc
}

void *size_attributed_malloc(__SIZE_TYPE__) __attribute__((alloc_size(1))) _TYPED(typed_malloc, 1);
void *size_attributed_typed_malloc(__SIZE_TYPE__, unsigned long long) __attribute__((alloc_size(1)));
void *unattributed_malloc(__SIZE_TYPE__) _TYPED(size_attributed_typed_malloc, 1);


int attribute_test() {
  size_attributed_malloc(1); // #alloc19
  // CHECK: %call = call ptr @typed_malloc(i64 noundef 1, i64 noundef [[LOC4_DESC:[0-9]+]])
  size_attributed_typed_malloc(2, 0); // #alloc20
  // CHECK: %call1 = call ptr @size_attributed_typed_malloc(i64 noundef 2, i64 noundef 0) [[ATTR:#[0-9]+]]
  unattributed_malloc(3); // #alloc21
  // CHECK: %call2 = call ptr @size_attributed_typed_malloc(i64 noundef 3, i64 noundef [[LOC5_DESC:[0-9]+]]) [[ATTR]]
  typed_malloc(4, 0); // #alloc22
  // CHECK: %call3 = call ptr @typed_malloc(i64 noundef 4, i64 noundef 0)
  malloc(4); // #alloc23
  // CHECK: %call4 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[LOC6_DESC:[0-9]+]])
  return 0;
}

int genericTest() {
  malloc(_Generic(0.5f, int: sizeof(struct S1), float: sizeof(struct S2))); // #alloc24
  // expected-remark@#alloc24 {{passing TMO information for type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc24 {{inferred 'struct S2' from expression '_Generic(0.5F, int: sizeof(struct S1), float: sizeof(struct S2))'}}
  // expected-note@#alloc24 {{encoding 'struct S2' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  malloc(_Generic(0, int: sizeof(struct S1), float: sizeof(struct S2))); // #alloc25
  // expected-remark@#alloc25 {{passing TMO information for type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc25 {{inferred 'struct S1' from expression '_Generic(0, int: sizeof(struct S1), float: sizeof(struct S2))'}}
  // expected-note@#alloc25 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  malloc(_Generic("foo", int: sizeof(struct S1), char*: 5)); // #alloc26
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 5, i64 noundef [[LOC7_DESC:[0-9]+]])
  malloc(_Generic("foo", int: 7, char*: sizeof(struct S2))); // #alloc27
  // expected-remark@#alloc27 {{passing TMO information for type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc27 {{inferred 'struct S2' from expression '_Generic("foo", int: 7, char *: sizeof(struct S2))'}}
  // expected-note@#alloc27 {{encoding 'struct S2' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call3 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  return 0;
}

void* f3(__SIZE_TYPE__ a);

void f3_test1() {
  f3(sizeof(int)); // #alloc28
  // rdar://102172111 (Handle use of TMO rewritten function used before the canonical TMO attribute is specified)
  // CHECK: %call = call ptr @f3
}

void* f3(__SIZE_TYPE__ a) _TYPED(typed_malloc, 1);

void f3_test2() {
  f3(sizeof(int)); // #alloc29
  // expected-remark@#alloc29 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'f3')}}
  // expected-note@#alloc29 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#alloc29 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
}

void test_expression_sizeof() {
  malloc(sizeof(5)); // #alloc30
  // expected-remark@#alloc30 {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc30 {{inferred 'int' from expression 'sizeof (5)'}}
  // expected-note@#alloc30 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call = call ptr @malloc
  malloc(sizeof(5 * sizeof(int))); // #alloc31
  // expected-remark@#alloc31 {{passing TMO information for type '__size_t' (aka 'unsigned long') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc31 {{inferred '__size_t' (aka 'unsigned long') from expression 'sizeof (5 * sizeof(int))'}}
  // expected-note@#alloc31 {{encoding '__size_t' (aka 'unsigned long') as 72057868919062295. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3227415 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[GENERICDATA64_DESC:72057868919062295]])
  // CHECK-DISABLED: %call1 = call ptr @malloc
  malloc(sizeof(sizeof(unsigned))); // #alloc32
  // expected-remark@#alloc32 {{passing TMO information for type '__size_t' (aka 'unsigned long') to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc32 {{inferred '__size_t' (aka 'unsigned long') from expression 'sizeof (sizeof(unsigned int))'}}
  // expected-note@#alloc32 {{encoding '__size_t' (aka 'unsigned long') as 72057868919062295. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3227415 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[GENERICDATA64_DESC]])
  // CHECK-DISABLED: %call2 = call ptr @malloc
  struct S1 s;
  (void)malloc(sizeof(s.fptr)); // #alloc33
  // expected-remark@#alloc33 {{passing TMO information for type 'void (*)()' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc33 {{inferred 'void (*)()' from expression 'sizeof (s.fptr)'}}
  // expected-note@#alloc33 {{encoding 'void (*)()' as 2252077784904504. { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3093312312 }}}
  // CHECK: %call3 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[ANONPTR_DESC:2252077784904504]])
  // CHECK-DISABLED: %call3 = call ptr @malloc
}

void test_conflicting_types() {
  struct S1* s = (struct S1*)malloc(sizeof(struct S2)); // #alloc34
  // expected-remark@#alloc34 {{passing TMO information for type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc34 {{inferred 'struct S2' from expression 'sizeof(struct S2)'}}
  // expected-note@#alloc34 {{encoding 'struct S2' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK-NOT: %call3 = call ptr @typed_real_malloc(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call3 = call ptr @malloc
}


typedef int __attribute__((vector_size(16))) ivector16;
typedef int __attribute__((vector_size(16))) ivector16_2;
ivector16 *typed_ivalloc(__SIZE_TYPE__, unsigned long long descriptor);
ivector16 *ivalloc1(__SIZE_TYPE__) _TYPED(typed_ivalloc, 1);
ivector16_2 *ivalloc2(__SIZE_TYPE__) _TYPED(typed_ivalloc, 1);

void test_sugared_types(void) {
  ivector16 *a = ivalloc1(sizeof(ivector16)); // #alloc35
  // expected-remark@#alloc35 {{passing TMO information for type 'ivector16' (vector of 4 'int' values) to 'typed_ivalloc' (retargeted from 'ivalloc1')}}
  // expected-note@#alloc35 {{inferred 'ivector16' (vector of 4 'int' values) from expression 'sizeof(ivector16)'}}
  // expected-note@#alloc35 {{encoding 'ivector16' (vector of 4 'int' values) as 72057870075255784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1159420904 }}}
  // CHECK: call ptr @typed_ivalloc(i64 noundef 16, i64 noundef [[GENERICDATA128_DESC:72057870075255784]])
  ivector16 *b = ivalloc2(sizeof(ivector16)); // #alloc36
  // expected-remark@#alloc36 {{passing TMO information for type 'ivector16' (vector of 4 'int' values) to 'typed_ivalloc' (retargeted from 'ivalloc2')}}
  // expected-note@#alloc36 {{inferred 'ivector16' (vector of 4 'int' values) from expression 'sizeof(ivector16)'}}
  // expected-note@#alloc36 {{encoding 'ivector16' (vector of 4 'int' values) as 72057870075255784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1159420904 }}}
  // CHECK: call ptr @typed_ivalloc(i64 noundef 16, i64 noundef [[GENERICDATA128_DESC]])
  ivector16_2 *c = ivalloc1(sizeof(ivector16)); // #alloc37
  // expected-remark@#alloc37 {{passing TMO information for type 'ivector16' (vector of 4 'int' values) to 'typed_ivalloc' (retargeted from 'ivalloc1')}}
  // expected-note@#alloc37 {{inferred 'ivector16' (vector of 4 'int' values) from expression 'sizeof(ivector16)'}}
  // expected-note@#alloc37 {{encoding 'ivector16' (vector of 4 'int' values) as 72057870075255784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1159420904 }}}
  // CHECK: call ptr @typed_ivalloc(i64 noundef 16, i64 noundef [[GENERICDATA128_DESC]])
  ivector16_2 *d = ivalloc2(sizeof(ivector16)); // #alloc38
  // expected-remark@#alloc38 {{passing TMO information for type 'ivector16' (vector of 4 'int' values) to 'typed_ivalloc' (retargeted from 'ivalloc2')}}
  // expected-note@#alloc38 {{inferred 'ivector16' (vector of 4 'int' values) from expression 'sizeof(ivector16)'}}
  // expected-note@#alloc38 {{encoding 'ivector16' (vector of 4 'int' values) as 72057870075255784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1159420904 }}}
  // CHECK: call ptr @typed_ivalloc(i64 noundef 16, i64 noundef [[GENERICDATA128_DESC]])
}

void test_explicit_cast_inference(void) {
  __SIZE_TYPE__ s;

  void *typed_malloc_test_cast(__SIZE_TYPE__ size, unsigned long long);
  void *malloc_test_cast(__SIZE_TYPE__ size) _TYPED(typed_malloc_test_cast, 1);

  // Check that we can associate an explicit cast
  int *i = (int *)malloc_test_cast(s); // #alloc39
  // expected-remark@#alloc39 {{passing TMO information for type 'int' to 'typed_malloc_test_cast' (retargeted from 'malloc_test_cast')}}
  // expected-note@#alloc39 {{inferred 'int' from cast of result from call to '(int *)malloc_test_cast(s)'}}
  // expected-note@#alloc39 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call = call ptr @typed_malloc_test_cast(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call = call ptr @malloc_test_cast

  // Check that a non-void cast always dominates a void cast
  int *i2 = (int *)(void *)malloc_test_cast(s); // #alloc40
  // expected-remark@#alloc40 {{passing TMO information for type 'int' to 'typed_malloc_test_cast' (retargeted from 'malloc_test_cast')}}
  // expected-note@#alloc40 {{inferred 'int' from cast of result from call to '(int *)(void *)malloc_test_cast(s)'}}
  // expected-note@#alloc40 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_malloc_test_cast(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call1 = call ptr @malloc_test_cast
  int *i3 = (void *)(int *)malloc_test_cast(s); // #alloc41
  // expected-remark@#alloc41 {{passing TMO information for type 'int' to 'typed_malloc_test_cast' (retargeted from 'malloc_test_cast')}}
  // expected-note@#alloc41 {{inferred 'int' from cast of result from call to '(int *)malloc_test_cast(s)'}}
  // expected-note@#alloc41 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call2 = call ptr @typed_malloc_test_cast(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call2 = call ptr @malloc_test_cast

  // Check that we can associate casts across GNU statement expressions
  void *vp_s1 = (struct S1 *)({ // #castLocation
    __SIZE_TYPE__ s2 = s;
    malloc_test_cast(s2); // #alloc42
  });
  // expected-remark@#alloc42 {{passing TMO information for type 'struct S1' to 'typed_malloc_test_cast' (retargeted from 'malloc_test_cast')}}
  // expected-note@#castLocation {{inferred 'struct S1' from cast of result from call to '(struct S1 *)({}}
  // expected-note@#alloc42 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call3 = call ptr @typed_malloc_test_cast(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call3 = call ptr @malloc_test_cast

  // Check that we don't associate non-pointer casts
  __UINTPTR_TYPE__ *uptr_true = malloc_test_cast(sizeof(__UINTPTR_TYPE__)); // #alloc43
  // expected-remark@#alloc43 {{passing TMO information for type 'unsigned long' to 'typed_malloc_test_cast' (retargeted from 'malloc_test_cast')}}
  // expected-note@#alloc43 {{inferred 'unsigned long' from expression 'sizeof(unsigned long)'}}
  // expected-note@#alloc43 {{encoding 'unsigned long' as 72057868919062295. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3227415 }}}
  // CHECK: %call4 = call ptr @typed_malloc_test_cast(i64 noundef [[LOC:[0-9]+]], i64 noundef [[GENERICDATA64_DESC]])
  // CHECK-DISABLED: %call4 = call ptr @malloc_test_cast
  __UINTPTR_TYPE__ uptr0 = (__UINTPTR_TYPE__)malloc_test_cast(s); // #alloc44
  // CHECK-NOT: %call5 = call ptr @typed_malloc_test_cast(i64 noundef %[[LOC:[0-9]+]], i64 noundef [[GENERICDATA64_DESC]])
  // CHECK-DISABLED: %call5 = call ptr @malloc_test_cast
  __UINTPTR_TYPE__ uptr1 = (__UINTPTR_TYPE__)malloc_test_cast(s); // #alloc45
  // CHECK-NOT: %call6 = call ptr @typed_malloc_test_cast(i64 noundef [[LOC:[0-9]+]], i64 noundef [[GENERICDATA64_DESC]])
  // CHECK-DISABLED: %call6 = call ptr @malloc_test_cast

  // Check that we don't associate casts to interfaces returning a non-pointer type
  posix_memalign(&vp_s1, 0, sizeof(int)); // #alloc46
  // expected-remark@#alloc46 {{passing TMO information for type 'int' to 'typed_real_posix_memalign' (retargeted from 'posix_memalign')}}
  // expected-note@#alloc46 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#alloc46 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call7 = call i32 @typed_real_posix_memalign(ptr noundef %vp_s1, i64 noundef 0, i64 noundef [[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call7 = call i32 @posix_memalign

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wint-to-pointer-cast"
  #pragma clang diagnostic ignored "-Wunused-value"
  (int *)posix_memalign(&vp_s1, 0, s); // #alloc47
  #pragma clang diagnostic pop

  // CHECK-NOT: %call8 = call i32 @typed_real_posix_memalign(ptr noundef %vp_s1, i64 noundef 0, i64 noundef [[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call8 = call i32 @posix_memalign

  // Check we don't pick up unrelated explicit casts
  void *func(void *p);
  int *r0 = (int *)func(malloc_test_cast(s)); // #alloc48
  // CHECK-NOT: %call9 = call ptr @typed_malloc_test_cast(i64 noundef [[LOC:[0-9]+]], i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call9 = call ptr @malloc_test_cast
}

void test_explicit_cast_inference_perf() {
  __SIZE_TYPE__ s;
  #define A1                     \
    do {                         \
      int *p = (int *)malloc(s); \
    } while (0)
  #define A4 A1; A1; A1; A1
  #define A16 A4; A4; A4; A4
  #define A64 A16; A16; A16; A16
  #define A256 A64; A64; A64; A64
  #define A1024 A256; A256; A256; A256
  #define A4096 A1024; A1024; A1024; A1024
  #define A16384 A4096; A4096; A4096; A4096

  #ifdef TMO_REMARKS
  A64; // #alloc49
  #else
  A16384;
  #endif
  // expected-remark@#alloc49 64 {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#alloc49 64 {{inferred 'int' from cast of result from call to '(int *)malloc(s)'}}
  // expected-note@#alloc49 64 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
}

void test_assignment_expressions() {
  __SIZE_TYPE__ sz;
  void *ptr;

  // Simple assignment - should get struct S1 type descriptor
  malloc((sz = sizeof(struct S1))); //#tae_assign
  // expected-remark@#tae_assign {{passing TMO information for type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_assign {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_assign {{inferred 'struct S1' from expression 'sz = sizeof(struct S1)'}}
  // CHECK: call ptr @typed_real_malloc(i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: call ptr @malloc(i64 noundef 24)

  // Nested assignment - should get int type descriptor
  __SIZE_TYPE__ sz2;
  malloc((sz = sz2 = sizeof(int))); // #tae_nestedassign
  // expected-remark@#tae_nestedassign {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_nestedassign {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_nestedassign {{inferred 'int' from expression 'sz = sz2 = sizeof(int)'}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 noundef 4, i64 noundef [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: call ptr @malloc(i64 noundef 4)

  // Compound assignment - should get int type descriptor
  sz = 0;
  malloc((sz += sizeof(int))); // #tae_addassign
  // expected-remark@#tae_addassign {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_addassign {{encoding 'int' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_addassign {{inferred indeterminate set of {'int'} from expression 'sz += sizeof(int)'}}
  // FIXME: it's not clear why we
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 noundef %add, i64 noundef [[INDETERMINATE_INT:72057595422605840]])
  // CHECK-DISABLED: call ptr @malloc
  // Equivalent non-compound form of += above - should also get int type descriptor
  sz = 0;
  malloc((sz = sz + sizeof(int))); // #tae_add
  // expected-remark@#tae_add {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_add {{encoding 'int' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_add {{inferred indeterminate set of {'int'} from expression 'sz = sz + sizeof(int)'}}
  // CHECK: %call4 = call ptr @typed_real_malloc(i64 noundef %add3, i64 noundef [[INDETERMINATE_INT]])
  // CHECK-DISABLED: call ptr @malloc
  // Compound assignment with *= - should get struct S2 type descriptor
  sz = 10;
  malloc((sz *= sizeof(struct S2))); // #tae_mulassign
  // expected-remark@#tae_mulassign {{passing TMO information for array of type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_mulassign {{encoding array of 'struct S2' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_mulassign {{inferred array of 'struct S2' from expression 'sz *= sizeof(struct S2)'}}
  // CHECK: %call5 = call ptr @typed_real_malloc(i64 noundef %mul, i64 noundef [[ARRAY_INT32_DESC]])
  // CHECK-DISABLED: call ptr @malloc

  // Equivalent non-compound form of *= above - should also get struct S2 type descriptor
  sz = 10;
  malloc((sz = sz * sizeof(struct S2))); // #tae_mul
  // expected-remark@#tae_mul {{passing TMO information for array of type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_mul {{encoding array of 'struct S2' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_mul {{inferred array of 'struct S2' from expression 'sz = sz * sizeof(struct S2)'}}
  // CHECK: %call7 = call ptr @typed_real_malloc(i64 noundef %mul6, i64 noundef [[ARRAY_INT32_DESC]])
  // CHECK-DISABLED: call ptr @malloc

  sz = 72;
  malloc((sz /= sizeof(int))); // #tae_divassign
  // expected-remark@#tae_divassign {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_divassign {{encoding 'int' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_divassign {{inferred indeterminate set of {'int'} from expression 'sz /= sizeof(int)'}}
  // CHECK: %call8 = call ptr @typed_real_malloc(i64 noundef %div, i64 noundef [[INDETERMINATE_INT]])
  // CHECK-DISABLED: call ptr @malloc

  sz = 72;
  malloc((sz -= sizeof(int))); // #tae_subassign
  // expected-remark@#tae_subassign {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_subassign {{encoding 'int' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_subassign {{inferred indeterminate set of {'int'} from expression 'sz -= sizeof(int)'}}
  // CHECK: %call9 = call ptr @typed_real_malloc(i64 noundef %sub, i64 noundef [[INDETERMINATE_INT]])
  // CHECK-DISABLED: call ptr @malloc

  // Assignment with multiplication (verify no regression)
  malloc((sz = 2 * sizeof(struct S2))); // #tae_constmult
  // expected-remark@#tae_constmult {{passing TMO information for array of type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_constmult {{encoding array of 'struct S2' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // expected-note@#tae_constmult {{inferred array of 'struct S2' from expression 'sz = 2 * sizeof(struct S2)'}}
  // CHECK: %call10 = call ptr @typed_real_malloc(i64 noundef 8, i64 noundef [[ARRAY_INT32_DESC]])
  // CHECK-DISABLED: call ptr @malloc(i64 noundef 8)

  // Test calloc with assignment in size parameter
  calloc(1, (sz = sizeof(struct S1))); // #tae_calloc
  // expected-remark@#tae_calloc {{passing TMO information for type 'struct S1' to 'typed_real_calloc' (retargeted from 'calloc')}}
  // expected-note@#tae_calloc {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_calloc {{inferred 'struct S1' from expression 'sz = sizeof(struct S1)'}}
  // CHECK: %call11 = call ptr @typed_real_calloc(i64 noundef 1, i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call11 = call ptr @calloc(i64 noundef 1, i64 noundef 24)

  // Test realloc with assignment - allocate initial memory then realloc larger
  ptr = malloc(2 * sizeof(struct S1)); // #tae_malloc_for_realloc
  // expected-remark@#tae_malloc_for_realloc {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#tae_malloc_for_realloc {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_malloc_for_realloc {{inferred array of 'struct S1' from expression '2 * sizeof(struct S1)'}}

  realloc(ptr, (sz = 10 * sizeof(struct S1))); // #tae_realloc
  // expected-remark@#tae_realloc {{passing TMO information for array of type 'struct S1' to 'typed_real_realloc' (retargeted from 'realloc')}}
  // expected-note@#tae_realloc {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_realloc {{inferred array of 'struct S1' from expression 'sz = 10 * sizeof(struct S1)'}}
  // CHECK: %call13 = call ptr @typed_real_realloc(ptr noundef %{{[0-9]+}}, i64 noundef 240, i64 noundef [[ARRAY_S1_DESC]])
  // CHECK-DISABLED: %call13 = call ptr @realloc(ptr noundef %{{[0-9]+}}, i64 noundef 240)

  // Test aligned_alloc with assignment
  aligned_alloc(16, (sz = sizeof(struct S1))); // #tae_aligned_alloc
  // expected-remark@#tae_aligned_alloc {{passing TMO information for type 'struct S1' to 'typed_real_aligned_alloc' (retargeted from 'aligned_alloc')}}
  // expected-note@#tae_aligned_alloc {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_aligned_alloc {{inferred 'struct S1' from expression 'sz = sizeof(struct S1)'}}
  // CHECK: %call14 = call ptr @typed_real_aligned_alloc(i64 noundef 16, i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call14 = call align 16 ptr @aligned_alloc(i64 noundef 16, i64 noundef 24)

  // Test posix_memalign with assignment
  void *memptr;
  posix_memalign(&memptr, 16, (sz = sizeof(struct S1))); // #tae_memalign
  // expected-remark@#tae_memalign {{passing TMO information for type 'struct S1' to 'typed_real_posix_memalign' (retargeted from 'posix_memalign')}}
  // expected-note@#tae_memalign {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // expected-note@#tae_memalign {{inferred 'struct S1' from expression 'sz = sizeof(struct S1)'}}
  // CHECK: %call15 = call i32 @typed_real_posix_memalign(ptr noundef %memptr, i64 noundef 16, i64 noundef 24, i64 noundef [[S1_DESC]])
  // CHECK-DISABLED: %call15 = call i32 @posix_memalign(ptr noundef %memptr, i64 noundef 16, i64 noundef 24)
}

void test_size_vs_cast(void) {
  __SIZE_TYPE__ n;

  struct S1 *p1 = (struct S1 *)malloc(sizeof(char) * n); // #cast_dom1
  // expected-remark@#cast_dom1 {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#cast_dom1 {{inferred array of 'struct S1' from cast of result from call to '(struct S1 *)malloc(sizeof(char) * n)'}}
  // expected-note@#cast_dom1 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}

  struct S2 *p2 = (struct S2 *)malloc(sizeof(struct S1) * n); // #size_dom1
  // expected-remark@#size_dom1 {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#size_dom1 {{inferred array of 'struct S1' from expression 'sizeof(struct S1) * n'}}
  // expected-note@#size_dom1 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}

  void *p3 = (void *)(struct S1 *)malloc(sizeof(char) * n); // #cast_dom2
  // expected-remark@#cast_dom2 {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#cast_dom2 {{inferred array of 'struct S1' from cast of result from call to '(struct S1 *)malloc(sizeof(char) * n)'}}
  // expected-note@#cast_dom2 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}

  void *p4 = (struct S1 *)(void *)malloc(sizeof(char) * n); // #cast_dom3
  // expected-remark@#cast_dom3 {{passing TMO information for array of type 'struct S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#cast_dom3 {{inferred array of 'struct S1' from cast of result from call to '(struct S1 *)(void *)malloc(sizeof(char) * n)'}}
  // expected-note@#cast_dom3 {{encoding array of 'struct S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}

  void *p5 = (struct S1 *)malloc(sizeof(struct S2) * n); // #size_dom2
  // expected-remark@#size_dom2 {{passing TMO information for array of type 'struct S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@#size_dom2 {{inferred array of 'struct S2' from expression 'sizeof(struct S2) * n'}}
  // expected-note@#size_dom2 {{encoding array of 'struct S2' as 72058145178419728}}
}

// CHECK: attributes [[ATTR]] = { allocsize(0) }

// CHECK: !{!"type-descriptor", !"[[LOC1_DESC]]", !"[[LOC1_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[LOC2_DESC]]", !"[[LOC2_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[LOC3_DESC]]", !"[[LOC3_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_INT32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_DOUBLE_DESC]]", !"3227415", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[U2_DESC]]", !"594631182", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ \22HasMixedUnions\22 ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"} 
// CHECK: !{!"type-descriptor", !"[[LOC4_DESC]]", !"[[LOC4_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[LOC5_DESC]]", !"[[LOC5_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[LOC6_DESC]]", !"[[LOC6_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[LOC7_DESC]]", !"[[LOC7_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA64_DESC]]", !"3227415", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ANONPTR_DESC]]", !"3093312312", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA128_DESC]]", !"1159420904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[INDETERMINATE_INT]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
