// RUN: %clang_cc1 -Rtmo-remarks -fsyntax-only -verify=tmo \
// RUN:               -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1 -Rtmo-remarks -fsyntax-only -verify=tmo,tmowarn -Wtyped-memory-inference-failure \
// RUN:               -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1 -Rtmo-remarks -fsyntax-only -verify=tmo \
// RUN:               -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1 -Rtmo-remarks -fsyntax-only -verify=tmo,tmowarn -Wtyped-memory-inference-failure \
// RUN:               -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1               -fsyntax-only -verify=notmoremarks \
// RUN:               -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1               -verify=notmoremarks -fsyntax-only \
// RUN:             -fno-typed-memory-operations -Rtmo-remarks -DTMO=0 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis %s
// RUN: %clang_cc1    -ftyped-memory-operations -DTMO=1 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis -emit-llvm -o - %s | FileCheck --check-prefix=CHECK          %s
// RUN: %clang_cc1 -fno-typed-memory-operations -DTMO=0 -nostdsysteminc -O0 -disable-llvm-passes -no-enable-noundef-analysis -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DISABLED %s

// notmoremarks-no-diagnostics
#if TMO
static_assert(__has_feature(typed_memory_operations));
#else
static_assert(!__has_feature(typed_memory_operations));
#endif

#define TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *typed_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size, unsigned long long);
void typed_unusual_alloc(float, unsigned, unsigned long long, const char*);

void *my_malloc(__SIZE_TYPE__ size) TYPED(typed_malloc, 1);
void *my_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size) TYPED(typed_calloc, 2);
void unusual_alloc(float, unsigned, const char*) TYPED(typed_unusual_alloc, 2);

extern "C" {

void *typed_real_malloc(__SIZE_TYPE__ size, unsigned long long);
void *typed_real_calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size, unsigned long long);
void *typed_real_realloc(void *, __SIZE_TYPE__ size, unsigned long long);
void *typed_real_aligned_alloc(unsigned long long align, unsigned long long size, unsigned long long);
int typed_real_posix_memalign(void **, __SIZE_TYPE__, __SIZE_TYPE__, unsigned long long);

void *malloc(__SIZE_TYPE__ size) TYPED(typed_real_malloc, 1);
void *calloc(__SIZE_TYPE__ count, __SIZE_TYPE__ size) TYPED(typed_real_calloc, 2);
void *realloc(void *, __SIZE_TYPE__ size) TYPED(typed_real_realloc, 2);
void *aligned_alloc(unsigned long long align, unsigned long long size) TYPED(typed_real_aligned_alloc, 2);
int posix_memalign(void **, __SIZE_TYPE__, __SIZE_TYPE__) TYPED(typed_real_posix_memalign, 3);
}


struct S1 {
  void *p;
  int i;
  int j;
  void (*fptr)();
};

struct S2 : S1 {

};

struct S3 {
  virtual ~S3();
};

struct S4 : S3 {

};

struct S4_2 : S3 {
  int i;
};

struct S5 {
  int i;
};

struct S6 : S5 {
  void *p;
};

struct S7 : virtual S5 {
  void *p;
};
struct S8 : virtual S5 {
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
extern "C" int f1();
extern "C" int f2();

extern "C" void f();
extern "C" void f() {
  // Basic codegen tests
  malloc(5); // #alloc1
  // tmowarn-warning@#alloc1 {{could not infer allocation type in call to 'malloc'}}
  // tmowarn-note@#alloc1 {{unable to infer allocation type from expression '5'}}
  // tmo@alloc1 {{unable to infer allocation type from expression '5'}}
  // CHECK: %call = call ptr @typed_real_malloc(i64 5, i64 [[LOC1_DESC:[0-9]+]])
  // CHECK-DISABLED: %call = call ptr @malloc(i64 5)
  malloc(sizeof(int)); // #alloc2
  // tmo-remark@#alloc2 {{passing TMO information for type 'int' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc2 {{inferred 'int' from expression 'sizeof(int)'}}
  // tmo-note@#alloc2 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call1 = call ptr @typed_real_malloc(i64 4, i64 [[GENERICDATA32_DESC:72057870300512784]])
  // CHECK-DISABLED: %call1 = call ptr @malloc(i64 4)
  malloc(sizeof(S1)); // #alloc3
  // tmo-remark@#alloc3 {{passing TMO information for type 'S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc3 {{inferred 'S1' from expression 'sizeof(S1)'}}
  // tmo-note@#alloc3 {{encoding 'S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call2 = call ptr @typed_real_malloc(i64 24, i64 [[S1_DESC:74309672738655766]])
  // CHECK-DISABLED: %call2 = call ptr @malloc(i64 24)
  malloc(sizeof(S1) * 5); // #alloc4
  // tmo-remark@#alloc4 {{passing TMO information for array of type 'S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc4 {{inferred array of 'S1' from expression 'sizeof(S1) * 5'}}
  // tmo-note@#alloc4 {{encoding array of 'S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call3 = call ptr @typed_real_malloc(i64 120, i64 [[ARRAY_S1_DESC:74309947616562710]])
  // CHECK-DISABLED: %call3 = call ptr @malloc(i64 120)
  malloc(sizeof(S1) + sizeof(S2)); // #alloc5
  // tmo-remark@#alloc5 {{passing TMO information for type 'S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc5 {{inferred tuple of ('S1', 'S2') from expression 'sizeof(S1) + sizeof(S2)'}}
  // tmo-note@#alloc5 {{encoding 'S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call4 = call ptr @typed_real_malloc(i64 48, i64 [[S1_DESC]])
  // CHECK-DISABLED: %call4 = call ptr @malloc(i64 48)
  my_malloc(5); // #alloc6
  // tmowarn-warning@#alloc6 {{could not infer allocation type in call to 'my_malloc'}}
  // tmowarn-note@#alloc6 {{unable to infer allocation type from expression '5'}}
  // tmo@alloc6 {{unable to infer allocation type from expression '5'}}
  // CHECK: %call5 = call ptr @_Z12typed_mallocmy(i64 5, i64 [[LOC2_DESC:[0-9]+]])
  // CHECK-DISABLED: %call5 = call ptr @_Z9my_mallocm(i64 5)
  my_malloc(sizeof(int)); // #alloc7
  // tmo-remark@#alloc7 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc7 {{inferred 'int' from expression 'sizeof(int)'}}
  // tmo-note@#alloc7 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call6 = call ptr @_Z12typed_mallocmy(i64 4, i64 [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call6 = call ptr @_Z9my_mallocm(i64 4)
  my_malloc(sizeof(S1)); // #alloc8
  // tmo-remark@#alloc8 {{passing TMO information for type 'S1' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc8 {{inferred 'S1' from expression 'sizeof(S1)'}}
  // tmo-note@#alloc8 {{encoding 'S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call7 = call ptr @_Z12typed_mallocmy(i64 24, i64 [[S1_DESC]])
  // CHECK-DISABLED: %call7 = call ptr @_Z9my_mallocm(i64 24)
  my_malloc(sizeof(S1) * 5); // #alloc9
  // tmo-remark@#alloc9 {{passing TMO information for array of type 'S1' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc9 {{inferred array of 'S1' from expression 'sizeof(S1) * 5'}}
  // tmo-note@#alloc9{{encoding array of 'S1' as 74309947616562710. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call8 = call ptr @_Z12typed_mallocmy(i64 120, i64 [[ARRAY_S1_DESC]])
  // CHECK-DISABLED: %call8 = call ptr @_Z9my_mallocm(i64 120)
  my_malloc(sizeof(S1) + sizeof(U1)); // #alloc10
  // tmo-remark@#alloc10 {{passing TMO information for type 'S1' to 'typed_malloc' (retargeted from 'my_malloc')}}
  // tmo-note@#alloc10 {{inferred tuple of ('S1', 'U1') from expression 'sizeof(S1) + sizeof(U1)'}}
  // tmo-note@#alloc10 {{encoding 'S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call9 = call ptr @_Z12typed_mallocmy(i64 28, i64 [[S1_DESC]])
  // CHECK-DISABLED: %call9 = call ptr @_Z9my_mallocm(i64 28)

  // Ordering of argument evaluation
  my_calloc(f1(), f2()); // #alloc11
  // tmowarn-warning@#alloc11 {{could not infer allocation type in call to 'my_calloc'}}
  // tmowarn-note@#alloc11 {{unable to infer allocation type from expression 'f2()'}}
  // CHECK: %call10 = call i32 @f1()
  // CHECK: %conv = sext i32 %call10 to i64
  // CHECK: %call11 = call i32 @f2()
  // CHECK: %conv12 = sext i32 %call11 to i64
  // CHECK: %call13 = call ptr @_Z12typed_callocmmy(i64 %conv, i64 %conv12, i64 [[LOC3_DESC:[0-9]+]])
  // CHECK-DISABLED: %call13 = call ptr @_Z9my_callocmm
  my_calloc(f1(), sizeof(int) * f2()); // #alloc12
  // tmo-remark@#alloc12 {{passing TMO information for array of type 'int' to 'typed_calloc' (retargeted from 'my_calloc')}}
  // tmo-note@#alloc12 {{inferred array of 'int' from expression 'sizeof(int) * f2()'}}
  // tmo-note@#alloc12 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call14 = call i32 @f1()
  // CHECK: %conv15 = sext i32 %call14 to i64
  // CHECK: %call16 = call i32 @f2()
  // CHECK: %conv17 = sext i32 %call16 to i64
  // CHECK: %mul = mul i64 4, %conv17
  // CHECK: %call18 = call ptr @_Z12typed_callocmmy(i64 %conv15, i64 %mul, i64 [[ARRAY_INT32_DESC:72058145178419728]])
  // CHECK-DISABLED: %call18 = call ptr @_Z9my_callocmm
  my_calloc(f1(), f2() * sizeof(double)); // #alloc13
  // tmo-remark@#alloc13 {{passing TMO information for array of type 'double' to 'typed_calloc' (retargeted from 'my_calloc')}}
  // tmo-note@#alloc13 {{inferred array of 'double' from expression 'f2() * sizeof(double)'}}
  // tmo-note@#alloc13 {{encoding array of 'double' as 72058143796969239. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 3227415 }}}
  // CHECK: %call19 = call i32 @f1()
  // CHECK: %conv20 = sext i32 %call19 to i64
  // CHECK: %call21 = call i32 @f2()
  // CHECK: %conv22 = sext i32 %call21 to i64
  // CHECK: %mul23 = mul i64 %conv22, 8
  // CHECK: %call24 = call ptr @_Z12typed_callocmmy(i64 %conv20, i64 %mul23, i64 [[ARRAY_DOUBLE_DESC:72058143796969239]])
  // CHECK-DISABLED: %call24 = call ptr @_Z9my_callocmm

  // Order of arguments passed to target
  unusual_alloc(0.5, sizeof(int) * f1(), "womp"); // #alloc14
  // tmo-remark@#alloc14 {{passing TMO information for array of type 'int' to 'typed_unusual_alloc' (retargeted from 'unusual_alloc')}}
  // tmo-note@#alloc14 {{inferred array of 'int' from expression 'sizeof(int) * f1()'}}
  // tmo-note@#alloc14 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call25 = call i32 @f1()
  // CHECK: %conv26 = sext i32 %call25 to i64
  // CHECK: %mul27 = mul i64 4, %conv26
  // CHECK: %conv28 = trunc i64 %mul27 to i32
  // CHECK: call void @_Z19typed_unusual_allocfjyPKc(float 5.000000e-01, i32 %conv28, i64 [[ARRAY_INT32_DESC]], ptr @.str)
  // CHECK-DISABLED: call void @_Z13unusual_allocfjPKc

 
  malloc(sizeof(S3)); // #alloc15
  // tmo-remark@#alloc15 {{passing TMO information for type 'S3' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc15 {{inferred 'S3' from expression 'sizeof(S3)'}}
  // tmo-note@#alloc15 {{encoding 'S3' as 2269669970948920. { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ "IsPolymorphic" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3093312312 }}}
  // CHECK: %call29 = call ptr @typed_real_malloc(i64 8, i64 [[S3_DESC:2269669970948920]])
  // CHECK-DISABLED: %call29 = call ptr @malloc
  malloc(sizeof(S4)); // #alloc16
  // tmo-remark@#alloc16 {{passing TMO information for type 'S4' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc16 {{inferred 'S4' from expression 'sizeof(S4)'}}
  // tmo-note@#alloc16 {{encoding 'S4' as 2269669970948920. { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ "IsPolymorphic" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3093312312 }}}
  // CHECK: %call30 = call ptr @typed_real_malloc(i64 8, i64 [[S3_DESC]])
  // CHECK-DISABLED: %call30 = call ptr @malloc
  malloc(sizeof(S5)); // #alloc17
  // tmo-remark@#alloc17 {{passing TMO information for type 'S5' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc17 {{inferred 'S5' from expression 'sizeof(S5)'}}
  // tmo-note@#alloc17 {{encoding 'S5' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call31 = call ptr @typed_real_malloc(i64 4, i64 [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call31 = call ptr @malloc
  malloc(sizeof(S6)); // #alloc18
  // tmo-remark@#alloc18 {{passing TMO information for type 'S6' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc18 {{inferred 'S6' from expression 'sizeof(S6)'}}
  // tmo-note@#alloc18 {{encoding 'S6' as 74309670676837506. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1947317378 }}}
  // CHECK: %call32 = call ptr @typed_real_malloc(i64 16, i64 [[S6_DESC:74309670676837506]])
  // CHECK-DISABLED: %call32 = call ptr @malloc
  malloc(sizeof(S7)); // #alloc19
  // tmo-remark@#alloc19 {{passing TMO information for type 'S7' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc19 {{inferred 'S7' from expression 'sizeof(S7)'}}
  // tmo-note@#alloc19 {{encoding 'S7' as 74309670376583829. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1647063701 }}}
  // CHECK: %call33 = call ptr @typed_real_malloc(i64 24, i64 [[S7_DESC:74309670376583829]])
  // CHECK-DISABLED: %call33 = call ptr @malloc
  malloc(sizeof(S8)); // #alloc20
  // tmo-remark@#alloc20 {{passing TMO information for type 'S8' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc20 {{inferred 'S8' from expression 'sizeof(S8)'}}
  // tmo-note@#alloc20 {{encoding 'S8' as 74309671181593780. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 2452073652 }}}
  // CHECK: %call34 = call ptr @typed_real_malloc(i64 16, i64 [[S8_DESC:74309671181593780]])
  // CHECK-DISABLED: %call34 = call ptr @malloc
  malloc(sizeof(U1)); // #alloc21
  // tmo-remark@#alloc21 {{passing TMO information for type 'U1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc21 {{inferred 'U1' from expression 'sizeof(U1)'}}
  // tmo-note@#alloc21 {{encoding 'U1' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  // CHECK: %call35 = call ptr @typed_real_malloc(i64 4, i64 [[GENERICDATA32_DESC]])
  // CHECK-DISABLED: %call35 = call ptr @malloc
  malloc(sizeof(U2)); // #alloc22
  // tmo-remark@#alloc22 {{passing TMO information for type 'U2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc22 {{inferred 'U2' from expression 'sizeof(U2)'}}
  // tmo-note@#alloc22 {{encoding 'U2' as 74344853696240142. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 594631182 }}}
  // CHECK: %call36 = call ptr @typed_real_malloc(i64 8, i64 [[U2_DESC:74344853696240142]])
  // CHECK-DISABLED: %call36 = call ptr @malloc
  malloc(sizeof(S1)); // #alloc23
  // tmo-remark@#alloc23 {{passing TMO information for type 'S1' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc23 {{inferred 'S1' from expression 'sizeof(S1)'}}
  // tmo-note@#alloc23 {{encoding 'S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK: %call37 = call ptr @typed_real_malloc(i64 24, i64 [[S1_DESC]])
  // CHECK-DISABLED: %call37 = call ptr @malloc
  malloc(sizeof(S2)); // #alloc24
  // tmo-remark@#alloc24 {{passing TMO information for type 'S2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc24 {{inferred 'S2' from expression 'sizeof(S2)'}}
  // tmo-note@#alloc24 {{encoding 'S2' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  // CHECK-DISABLED: %call38 = call ptr @malloc
  // CHECK: %call38 = call ptr @typed_real_malloc(i64 24, i64 [[S1_DESC]])
  malloc(sizeof(S3)); // #alloc25
  // tmo-remark@#alloc25 {{passing TMO information for type 'S3' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc25 {{inferred 'S3' from expression 'sizeof(S3)'}}
  // tmo-note@#alloc25 {{encoding 'S3' as 2269669970948920. { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ "IsPolymorphic" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3093312312 }}}
  // CHECK-DISABLED: %call39 = call ptr @malloc
  // CHECK: %call39 = call ptr @typed_real_malloc(i64 8, i64 [[S3_DESC]])
  malloc(sizeof(S4_2)); // #alloc26
  // tmo-remark@#alloc26 {{passing TMO information for type 'S4_2' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc26 {{inferred 'S4_2' from expression 'sizeof(S4_2)'}}
  // tmo-note@#alloc26 {{encoding 'S4_2' as 74327263367638196. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "IsPolymorphic" ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 2452073652 }}}
  // CHECK-DISABLED: %call40 = call ptr @malloc
  // CHECK: %call40 = call ptr @typed_real_malloc(i64 16, i64 [[S4_2_DESC:74327263367638196]])

  malloc(sizeof(nullptr)); // #alloc27
  // tmo-remark@#alloc27 {{passing TMO information for type 'std::nullptr_t' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc27 {{inferred 'std::nullptr_t' from expression 'sizeof (nullptr)'}}
  // tmo-note@#alloc27 {{encoding 'std::nullptr_t' as 74309671125801946. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 2396281818 }}}
  // CHECK-DISABLED: %call41 = call ptr @malloc
  // CHECK: %call41 = call ptr @typed_real_malloc(i64 8, i64 [[NULLPTR_DESC:74309671125801946]])

  // Unique enough type just to avoid conflating the descriptor with others
  struct NullptrStruct {
    decltype(nullptr) field1;
    int field2;
    double field3;
    char field4;
    double field5;
  };
  malloc(sizeof(NullptrStruct)); // #alloc28
  // tmo-remark@#alloc28 {{passing TMO information for type 'NullptrStruct' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // tmo-note@#alloc28 {{inferred 'NullptrStruct' from expression 'sizeof(NullptrStruct)'}}
  // tmo-note@#alloc28 {{encoding 'NullptrStruct' as 74309671971610988. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3242090860 }}}
  // CHECK-DISABLED: %call42 = call ptr @malloc
  // CHECK: %call42 = call ptr @typed_real_malloc(i64 40, i64 [[NULLPTRSTRUCT_DESC:74309671971610988]])
}

class C {

  struct I {
    int a;
    void *p;
    int b;
  };

  struct I *p;

public:
  void m() {
    __SIZE_TYPE__ sz;
    p = reinterpret_cast<struct I *>(malloc(sz)); // #alloc29
    // tmo-remark@#alloc29 {{passing TMO information for type 'struct I' to 'typed_real_malloc' (retargeted from 'malloc')}}
    // tmo-note@#alloc29 {{inferred 'struct I' from cast of result from call to 'reinterpret_cast<struct I *>(malloc(sz))'}}
    // tmo-note@#alloc29 {{encoding 'struct I' as 74309669678308648. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 948788520 }}}
    // CHECK-DISABLED: %call = call ptr @malloc(i64 %0)
    // CHECK: %call = call ptr @typed_real_malloc(i64 %0, i64 [[I_DESC:74309669678308648]])
  }
};

void g() {
  C().m();
}

// CHECK: !{!"type-descriptor", !"[[LOC1_DESC]]", !"[[LOC1_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[GENERICDATA32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[LOC2_DESC]]", !"[[LOC2_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[LOC3_DESC]]", !"[[LOC3_DESC]]", !"\22LayoutSemantics\22: [ ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_INT32_DESC]]", !"1384677904", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[ARRAY_DOUBLE_DESC]]", !"3227415", !"\22LayoutSemantics\22: [ \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22Array\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S3_DESC]]", !"3093312312", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22 ], \22TypeFlags\22: [ \22IsPolymorphic\22 ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S6_DESC]]", !"1947317378", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S7_DESC]]", !"1647063701", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S8_DESC]]", !"2452073652", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[U2_DESC]]", !"594631182", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ \22HasMixedUnions\22 ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S4_2_DESC]]", !"2452073652", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ \22IsPolymorphic\22 ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[NULLPTR_DESC]]", !"2396281818", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[NULLPTRSTRUCT_DESC]]", !"3242090860", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[I_DESC]]", !"948788520", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
