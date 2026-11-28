// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes \
// RUN:   -Wno-error=ignored-attributes -verify %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

enum AA { VAL1 = -100 };
enum e1 { E1_A = 1, E1_C = 10 } __reloc__;
typedef enum { E2_A = 3 } __reloc__ e2;
enum { E3_A = 4 } __reloc__; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
struct s {
  enum e4 { E4_A = 5 } a;
  enum e6 { E6_A = 6 } __reloc__ b;
} __reloc__;
enum s64 { S64_BIG = 0x100000000, S64_NEG = -5 } __reloc__;
enum u64 { U64_BIG = 0xffffffff80000000 } __reloc__;
enum e5 { E5_Z = 0 } __reloc__;

// CHECK: @[[E1_C:[0-9]+]] = private unnamed_addr constant [8 x i8] c"E1_C:10\00"
// CHECK: @[[E2_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E2_A:3\00"
// CHECK: @[[E6_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E6_A:6\00"
// CHECK: @[[S64_BIG:[0-9]+]] = private unnamed_addr constant [19 x i8] c"S64_BIG:4294967296\00"
// CHECK: @[[S64_NEG:[0-9]+]] = private unnamed_addr constant [11 x i8] c"S64_NEG:-5\00"
// CHECK: @[[U64_BIG:[0-9]+]] = private unnamed_addr constant [20 x i8] c"U64_BIG:-2147483648\00"

// With or without __builtin_preserve_access_index(), an enumerator is
// relocated. An enum in a struct does not get the struct's attribute, so E4_A
// is a constant, but E6_A, whose enum has its own attribute, is relocated.
int test1(int x) {
  return (x == E1_C) + __builtin_preserve_access_index(E2_A) + (E4_A) + E6_A;
}

// CHECK: define dso_local i32 @test1
// CHECK: [[X:%[0-9a-z]+]] = load i32, ptr %x.addr
// CHECK-NEXT: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E1_C]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1:[0-9]+]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: icmp eq i32 [[X]], [[T]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E2_A]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2:[0-9]+]]
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: add nsw i32 %{{[0-9a-z]+}}, 5
// CHECK-NEXT: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E6_A]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E6:[0-9]+]]

// A condition using an enumerator is not folded.
int test2(int x) {
  if (E1_C == 10)
    x++;
  return x;
}

// CHECK: define dso_local i32 @test2
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: icmp eq i32 [[T]], 10

// Local aggregates initialized with the enumerators are not constant
// initialized.
struct key { int a; int b; };
void use(void *);
void test3(void) {
  struct key k = { E1_A, 2 };
  long arr[4] = { 1, 2, 3, E1_C };
  use(&k);
  use(arr);
}

// CHECK: define dso_local void @test3
// CHECK-NOT: memcpy
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NOT: memcpy
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NOT: memcpy
// CHECK: ret void

// Enumerators of an enum without the attribute, of an anonymous enum without
// typedef name, and in unevaluated operands keep their compile-time value.
int test4(int x) {
  return x + VAL1 + E3_A + sizeof(E1_C);
}

// CHECK: define dso_local i32 @test4
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: add nsw i32 %{{[0-9a-z]+}}, -100
// CHECK-NEXT: add nsw i32 %{{[0-9a-z]+}}, 4
// CHECK: add i64 %{{[0-9a-z]+}}, 4
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32

// Enumerators with 64-bit values are relocated without truncation, since they
// have type long or unsigned long.
long test5(void) {
  return S64_BIG + S64_NEG;
}

// CHECK: define dso_local i64 @test5
// CHECK: [[V1:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[S64_BIG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[S64:[0-9]+]]
// CHECK: [[V2:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[S64_NEG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[S64]]
// CHECK: [[ADD:%[0-9a-z]+]] = add nsw i64 [[V1]], [[V2]]
// CHECK: ret i64 [[ADD]]

unsigned long test6(void) {
  return U64_BIG;
}

// CHECK: define dso_local i64 @test6
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[U64_BIG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[U64:[0-9]+]]
// CHECK: ret i64 [[V]]

// Local const variables initialized with the enumerators are not constant
// folded.
int test7(int x) {
  const int k = E1_C;
  const int k2 = k + 1;
  const struct key kk = { E1_A, 2 };
  if (k2 == 11)
    x++;
  return x + k + kk.a;
}

// CHECK: define dso_local i32 @test7
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK: store i32 [[T]], ptr %k,
// CHECK: [[K:%[0-9a-z]+]] = load i32, ptr %k,
// CHECK: add nsw i32 [[K]], 1
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: [[K2:%[0-9a-z]+]] = load i32, ptr %k2,
// CHECK: icmp eq i32 [[K2]], 11
// CHECK: load i32, ptr %k,
// CHECK: load i32, ptr %a{{[0-9]+}},
// CHECK: ret i32

// A conditional operator with relocated enumerators uses control flow, not a
// select, since the backend cannot relocate a select of two relocations.
int test8(int x) {
  return x ? E1_A : E1_C;
}

// CHECK: define dso_local i32 @test8
// CHECK: br i1 %{{[0-9a-z]+}}, label %cond.true, label %cond.false
// CHECK: cond.true:
// CHECK-NEXT: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: cond.false:
// CHECK-NEXT: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: cond.end:
// CHECK-NEXT: phi i32
// CHECK-NOT: select
// CHECK: ret i32

// Outside the argument of __builtin_preserve_enum_value(), an enumerator cast
// to a pointer to its enum is relocated like any other use. There is no
// -Wint-to-pointer-cast warning, as without the attribute (checked by -verify).
enum e1 *test9(void) {
  return (enum e1 *)E1_C;
}

// CHECK: define dso_local ptr @test9
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[S:%[0-9a-z]+]] = sext i32 [[T]] to i64
// CHECK-NEXT: inttoptr i64 [[S]] to ptr

// Only the enumerator in the argument of __builtin_preserve_enum_value() is
// not relocated, including an enumerator with value 0 and a typedef'd enum.
unsigned long test10(void) {
  return __builtin_preserve_enum_value(*(enum e5 *)E5_Z, 1) +
         __builtin_preserve_enum_value(*(e2 *)E2_A, 0) +
         (unsigned long)(void *)E5_Z;
}

// CHECK: define dso_local i64 @test10
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E5:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2]]
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E5]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[S:%[0-9a-z]+]] = sext i32 [[T]] to i64
// CHECK-NEXT: inttoptr i64 [[S]] to ptr
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i64

// A relocated enumerator is converted to its own type, here the enum type
// with a fixed underlying type, as without the attribute.
enum c8 : signed char { C8_NEG = -1 } __reloc__;
enum u8 : unsigned char { U8_A = 200 } __reloc__;

long test11(void) {
  return C8_NEG;
}

// CHECK: define dso_local i64 @test11
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[C8:[0-9]+]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i8
// CHECK-NEXT: [[S:%[0-9a-z]+]] = sext i8 [[T]] to i64
// CHECK-NEXT: ret i64 [[S]]

int test12(unsigned long x) {
  return x < U8_A;
}

// CHECK: define dso_local i32 @test12
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[U8:[0-9]+]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i8
// CHECK-NEXT: [[Z:%[0-9a-z]+]] = zext i8 [[T]] to i64
// CHECK-NEXT: icmp ult i64 %{{[0-9a-z]+}}, [[Z]]

// A relocated enumerator of type int keeps that type, so hash % E1_C is an
// unsigned 32-bit remainder, which does not need -mcpu=v4 as a signed one would.
unsigned test13(unsigned hash) {
  return hash % E1_C;
}

// CHECK: define dso_local i32 @test13
// CHECK: [[H:%[0-9a-z]+]] = load i32, ptr %hash.addr
// CHECK-NEXT: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: urem i32 [[H]], [[T]]

// In a nodebug function, there is no debug info to relocate against, so the
// value of the enumerator is used, as for a struct with the attribute.
__attribute__((nodebug)) int test14(int x) {
  return x + E1_C;
}

// CHECK: define dso_local i32 @test14
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: add nsw i32 %{{[0-9a-z]+}}, 10
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32

// A named enum and an anonymous enum with a typedef name get the attribute
// from #pragma clang attribute, and an enum gets it from a forward
// declaration, so their enumerators are relocated.
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = any(record, enum))
enum e7 { E7_A = 7 };
typedef enum { E8_A = 8 } e8;
#pragma clang attribute pop
enum __reloc__ e9;
enum e9 { E9_A = 9 };

int test15(int x) {
  return x + E7_A + E8_A + E9_A;
}

// CHECK: define dso_local i32 @test15
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E7:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E8:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E9:[0-9]+]]
// CHECK: ret i32

// The value of a relocated enumerator is not known at compile time, so
// __builtin_constant_p() of it is 0, unlike for an enum without the attribute.
int test16(void) {
  return __builtin_constant_p(E1_C) * 10 + __builtin_constant_p(VAL1);
}

// CHECK: define dso_local i32 @test16
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32 1

// CHECK-DAG: ![[E1]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e1"
// CHECK-DAG: ![[E2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "e2"
// CHECK-DAG: ![[E6]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e6"
// CHECK-DAG: ![[S64]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "s64"{{.*}}, size: 64,
// CHECK-DAG: ![[U64]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "u64"{{.*}}, size: 64,
// CHECK-DAG: ![[E5]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e5"
// CHECK-DAG: ![[C8]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "c8"
// CHECK-DAG: ![[U8]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "u8"
// CHECK-DAG: ![[E7]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e7"
// CHECK-DAG: ![[E8]] = !DIDerivedType(tag: DW_TAG_typedef, name: "e8"
// CHECK-DAG: ![[E9]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e9"
