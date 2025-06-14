// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm %s       -o- | FileCheck --check-prefixes=CHECK,CHECKC,CHECK128 %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -xc++ %s -o- | FileCheck --check-prefixes=CHECK,CHECK128 %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios    -fptrauth-calls -fptrauth-intrinsics -emit-pch %s -o %t.ast
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios    -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -emit-llvm -x ast -o - %t.ast | FileCheck --check-prefixes=CHECK,CHECKC,CHECK128 %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-disable-128bit-type-discrimination -disable-llvm-passes -emit-llvm %s       -o- | FileCheck --check-prefixes=CHECK,CHECKC,CHECKNO128 %s

// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm %s       -o- | FileCheck --check-prefixes=CHECK,CHECKC,CHECK128 %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -xc++ %s -o- | FileCheck --check-prefixes=CHECK,CHECK128 %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics -emit-pch %s -o %t.ast
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -emit-llvm -x ast -o - %t.ast | FileCheck --check-prefixes=CHECK,CHECKC,CHECK128 %s

#ifdef __cplusplus
extern "C" {
#endif

void f(void);
void f2(int);
void (*fnptr)(void);
void *opaque;
unsigned long uintptr;

// CHECK: @test_constant_null = global ptr null
void (*test_constant_null)(int) = 0;

// CHECK: @test_constant_cast = global ptr ptrauth (ptr @f, i32 0, i64 2712)
void (*test_constant_cast)(int) = (void (*)(int))f;

#ifndef __cplusplus
// CHECKC: @enum_func_ptr = global ptr ptrauth (ptr @enum_func, i32 0, i64 2712)
enum Enum0;
void enum_func(enum Enum0);
void (*enum_func_ptr)(enum Enum0) = enum_func;
#endif

// CHECK: @test_opaque = global ptr ptrauth (ptr @f, i32 0)
void *test_opaque =
#ifdef __cplusplus
    (void *)
#endif
    (void (*)(int))(double (*)(double))f;

// CHECK: @test_intptr_t = global i64 ptrtoint (ptr ptrauth (ptr @f, i32 0) to i64)
unsigned long test_intptr_t = (unsigned long)f;

// CHECK: @test_through_long = global ptr ptrauth (ptr @f, i32 0, i64 2712)
void (*test_through_long)(int) = (void (*)(int))(long)f;

// CHECK: @test_to_long = global i64 ptrtoint (ptr ptrauth (ptr @f, i32 0) to i64)
long test_to_long = (long)(double (*)())f;

extern void external_function(void);
// CHECK: @fptr1 = global ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*fptr1)(void) = external_function;
// CHECK: @fptr2 = global ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*fptr2)(void) = &external_function;

// CHECK: @fptr3 = global ptr ptrauth (ptr @external_function, i32 2, i64 26)
void (*fptr3)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, 26);

// CHECK: @fptr4 = global ptr ptrauth (ptr @external_function, i32 2, i64 26, ptr @fptr4)
void (*fptr4)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, __builtin_ptrauth_blend_discriminator(&fptr4, 26));

extern void external_function_int(int);
extern void external_function_char(char);
extern void external_function_i128(__int128_t);
extern void external_function_u128(__uint128_t);
extern void external_function_b128(_BitInt(128));
extern void external_function_b8(_BitInt(8));

// Check discriminators of functions taking integer type arguments:

//  - Builtin integer types should be discriminated equally (so, pointer to
//  function taking int argument should accept function taking char argument
//  - _BitInt types are guaranteed distinct and therefore should be discriminated
//  differently
//  - __int128_t / __uint128_t are passed differently than char / int / long
//  (require two registers instead of one) and therefore should be discriminated
//  differently.

// CHECK: @fptr5 = global ptr ptrauth (ptr @external_function_int, i32 0, i64 2712)
// CHECK: @fptr6 = global ptr ptrauth (ptr @external_function_char, i32 0, i64 2712)
void (*fptr5)(int) = external_function_int;
void (*fptr6)(char) = external_function_char;

// CHECK128: @fptr7 = global ptr ptrauth (ptr @external_function_i128, i32 0, i64 23141)
// CHECK128: @fptr8 = global ptr ptrauth (ptr @external_function_u128, i32 0, i64 45743)
// CHECK128: @fptr9 = global ptr ptrauth (ptr @external_function_b128, i32 0, i64 17854)
// CHECK128: @fptr10 = global ptr ptrauth (ptr @external_function_b8, i32 0, i64 26383)
// CHECKNO128: @fptr7 = global ptr ptrauth (ptr @external_function_i128, i32 0, i64 2712)
// CHECKNO128: @fptr8 = global ptr ptrauth (ptr @external_function_u128, i32 0, i64 2712)
// CHECKNO128: @fptr9 = global ptr ptrauth (ptr @external_function_b128, i32 0, i64 41228)
// CHECKNO128: @fptr10 = global ptr ptrauth (ptr @external_function_b8, i32 0, i64 41228)
void (*fptr7)(__int128_t) = external_function_i128;
void (*fptr8)(__uint128_t) = external_function_u128;
void (*fptr9)(_BitInt(128)) = external_function_b128;
void (*fptr10)(_BitInt(8)) = external_function_b8;

// CHECK-LABEL: define{{.*}} void @test_call()
void test_call() {
  // CHECK:      [[T0:%.*]] = load ptr, ptr @fnptr,
  // CHECK-NEXT: call void [[T0]]() [ "ptrauth"(i32 0, i64 18983) ]
  fnptr();
}

// CHECK-LABEL: define{{.*}} ptr @test_function_pointer()
// CHECK:  ret ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*test_function_pointer())(void) {
  return external_function;
}

struct InitiallyIncomplete;
extern struct InitiallyIncomplete returns_initially_incomplete(void);
// CHECK-LABEL: define{{.*}} void @use_while_incomplete()
void use_while_incomplete() {
  // CHECK:      [[VAR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr ptrauth (ptr @returns_initially_incomplete, i32 0, i64 25106), ptr [[VAR]]
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}
struct InitiallyIncomplete { int x; };
// CHECK-LABEL: define{{.*}} void @use_while_complete()
void use_while_complete() {
  // CHECK:      [[VAR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr ptrauth (ptr @returns_initially_incomplete, i32 0, i64 25106), ptr [[VAR]]
  // CHECK-NEXT: ret void
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}

#ifndef __cplusplus

void knr(param)
  int param;
{}

// CHECKC-LABEL: define{{.*}} void @test_knr
void test_knr() {
  void (*p)() = knr;
  p(0);

  // CHECKC: [[P:%.*]] = alloca ptr
  // CHECKC: store ptr ptrauth (ptr @knr, i32 0, i64 18983), ptr [[P]]
  // CHECKC: [[LOAD:%.*]] = load ptr, ptr [[P]]
  // CHECKC: call void [[LOAD]](i32 noundef 0) [ "ptrauth"(i32 0, i64 18983) ]
}

// CHECKC-LABEL: define{{.*}} void @test_redeclaration
void test_redeclaration() {
  void redecl();
  void (*ptr)() = redecl;
  void redecl(int);
  void (*ptr2)(int) = redecl;
  ptr();
  ptr2(0);

  // CHECKC: store ptr ptrauth (ptr @redecl, i32 0, i64 18983), ptr %ptr
  // CHECKC: store ptr ptrauth (ptr @redecl, i32 0, i64 2712), ptr %ptr2
  // CHECKC: call void {{.*}}() [ "ptrauth"(i32 0, i64 18983) ]
  // CHECKC: call void {{.*}}(i32 noundef 0) [ "ptrauth"(i32 0, i64 2712) ]
}

void knr2(param)
     int param;
{}

// CHECKC-LABEL: define{{.*}} void @test_redecl_knr
void test_redecl_knr() {
  void (*p)() = knr2;
  p();

  // CHECKC: store ptr ptrauth (ptr @knr2, i32 0, i64 18983)
  // CHECKC: call void {{.*}}() [ "ptrauth"(i32 0, i64 18983) ]

  void knr2(int);

  void (*p2)(int) = knr2;
  p2(0);

  // CHECKC: store ptr ptrauth (ptr @knr2, i32 0, i64 2712)
  // CHECKC: call void {{.*}}(i32 noundef 0) [ "ptrauth"(i32 0, i64 2712) ]
}

#endif

#ifdef __cplusplus
}
#endif
