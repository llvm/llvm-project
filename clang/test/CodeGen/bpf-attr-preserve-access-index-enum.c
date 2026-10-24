// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes \
// RUN:   -Wno-error=ignored-attributes -verify %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

enum AA { VAL1 = -100, VAL2 = 0xffff8000 };
typedef enum { VAL10 = 0xffffFFFF80000000 } __BB;
enum e1 { E1_A = 1, E1_C = 10 } __reloc__;
typedef enum { E2_A = 3 } __reloc__ e2;
enum { E3_A = 4 } __reloc__; // expected-warning {{attribute has no effect on an anonymous enum without a typedef name}}
struct s {
  enum e4 { E4_A = 5 } a;
} __reloc__;
enum s64 { S64_BIG = 0x100000000, S64_NEG = -5 } __reloc__;
enum u64 { U64_BIG = 0xffffffff80000000 } __reloc__;
enum e5 { E5_Z = 0 } __reloc__;

// CHECK: @[[VAL1:[0-9]+]] = private unnamed_addr constant [10 x i8] c"VAL1:-100\00"
// CHECK: @[[VAL2:[0-9]+]] = private unnamed_addr constant [16 x i8] c"VAL2:4294934528\00"
// CHECK: @[[VAL10:[0-9]+]] = private unnamed_addr constant [18 x i8] c"VAL10:-2147483648\00"
// CHECK: @[[E1_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E1_A:1\00"
// CHECK: @[[E1_C:[0-9]+]] = private unnamed_addr constant [8 x i8] c"E1_C:10\00"
// CHECK: @[[E2_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E2_A:3\00"
// CHECK: @[[E4_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E4_A:5\00"
// CHECK: @[[S64_BIG:[0-9]+]] = private unnamed_addr constant [19 x i8] c"S64_BIG:4294967296\00"
// CHECK: @[[S64_NEG:[0-9]+]] = private unnamed_addr constant [11 x i8] c"S64_NEG:-5\00"
// CHECK: @[[U64_BIG:[0-9]+]] = private unnamed_addr constant [20 x i8] c"U64_BIG:-2147483648\00"

// __builtin_preserve_enum_value() is not affected by the attribute.
int test1(void) {
  return __builtin_preserve_enum_value(*(enum AA *)VAL1, 0) +
         __builtin_preserve_enum_value(*(enum AA *)VAL2, 1) +
         __builtin_preserve_enum_value(*(__BB *)VAL10, 1) +
         __builtin_preserve_enum_value(*(enum e1 *)E1_A, 0);
}

// CHECK: define dso_local i32 @test1
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 0, ptr @[[VAL1]], i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[AA:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 1, ptr @[[VAL2]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[AA]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 2, ptr @[[VAL10]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[BB:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 3, ptr @[[E1_A]], i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1:[0-9]+]]
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32

// An enumerator of an attributed enum is an ENUM_VALUE relocation, with or
// without __builtin_preserve_access_index().
int test2(int x) {
  return (x == E1_C) + __builtin_preserve_access_index(E2_A) + (E4_A);
}

// CHECK: define dso_local i32 @test2
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E1_C]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: icmp eq i64 %{{[0-9a-z]+}}, [[V]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E2_A]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E4_A]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E4:[0-9]+]]

// Conditions and builtin calls using the enumerators are not folded. A
// relocated enumerator has type unsigned long, so x < E1_C is an unsigned
// compare.
int test3(int x) {
  if (E1_C == 10)
    x++;
  while (x < E1_C)
    x += E1_C > 5 ? 2 : 3;
  switch (E1_A) {
  case 1:
    x++;
    break;
  default:
    break;
  }
  return x + __builtin_bswap32(E1_C);
}

// CHECK: define dso_local i32 @test3
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: icmp eq i64 %{{[0-9a-z]+}}, 10
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: icmp ult i64
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: icmp ugt i64 %{{[0-9a-z]+}}, 5
// CHECK: select i1
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: switch i64 [[V]], label %{{.*}} [
// CHECK-NEXT: i64 1, label
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: call i32 @llvm.bswap.i32

// Local aggregates initialized with the enumerators are not constant
// initialized.
struct key { int a; int b; };
void use(void *);
void test4(void) {
  struct key k = { E1_A, 2 };
  long arr[4] = { 1, 2, 3, E1_C };
  use(&k);
  use(arr);
}

// CHECK: define dso_local void @test4
// CHECK-NOT: memcpy
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NOT: memcpy
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NOT: memcpy
// CHECK: ret void

// Enumerators of an enum without the attribute, of an anonymous enum without
// typedef name, and in unevaluated operands keep their compile-time value.
int test5(int x) {
  return x + VAL1 + E3_A + sizeof(E1_C);
}

// CHECK: define dso_local i32 @test5
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: add nsw i64 %{{[0-9a-z]+}}, -100
// CHECK-NEXT: add nsw i64 %{{[0-9a-z]+}}, 4
// CHECK-NEXT: add i64 %{{[0-9a-z]+}}, 4
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32

// Enumerators with 64-bit values are relocated without truncation.
long test6(void) {
  return S64_BIG + S64_NEG;
}

// CHECK: define dso_local i64 @test6
// CHECK: [[V1:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[S64_BIG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[S64:[0-9]+]]
// CHECK: [[V2:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[S64_NEG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[S64]]
// CHECK: [[ADD:%[0-9a-z]+]] = add i64 [[V1]], [[V2]]
// CHECK: ret i64 [[ADD]]

unsigned long test7(void) {
  return U64_BIG;
}

// CHECK: define dso_local i64 @test7
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[U64_BIG]], i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[U64:[0-9]+]]
// CHECK: ret i64 [[V]]

// Local const variables initialized with the enumerators are not constant
// folded.
int test8(int x) {
  const int k = E1_C;
  const int k2 = k + 1;
  const struct key kk = { E1_A, 2 };
  if (k2 == 11)
    x++;
  return x + k + kk.a;
}

// CHECK: define dso_local i32 @test8
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

// A self-referencing initializer is handled.
int test9(void) {
  const int k = k + E1_C;
  return k;
}

// CHECK: define dso_local i32 @test9
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: ret i32

// A conditional operator with relocated enumerators uses control flow, not a
// select, since the backend cannot relocate a select of two relocations.
int test10(int x) {
  return x ? E1_A : E1_C;
}

// CHECK: define dso_local i32 @test10
// CHECK: br i1 %{{[0-9a-z]+}}, label %cond.true, label %cond.false
// CHECK: cond.true:
// CHECK-NEXT: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: cond.false:
// CHECK-NEXT: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: cond.end:
// CHECK-NEXT: phi i64
// CHECK-NOT: select
// CHECK: ret i32

// A cast of an enumerator to a pointer to its enum is relocated as other
// uses, without a -Wint-to-pointer-cast warning since it has type unsigned
// long.
enum e1 *test11(void) {
  return (enum e1 *)E1_C;
}

// CHECK: define dso_local ptr @test11
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK: inttoptr i64 [[V]] to ptr

// Only the enumerator in the argument of __builtin_preserve_enum_value() is
// not relocated, including an enumerator with value 0 and a typedef'd enum.
unsigned long test12(void) {
  return __builtin_preserve_enum_value(*(enum e5 *)E5_Z, 1) +
         __builtin_preserve_enum_value(*(e2 *)E2_A, 0) +
         (unsigned long)(void *)E5_Z;
}

// CHECK: define dso_local i64 @test12
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E5:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2]]
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E5]]
// CHECK: inttoptr i64 [[V]] to ptr
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i64

// CHECK-DAG: ![[AA]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA"
// CHECK-DAG: ![[BB]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__BB"
// CHECK-DAG: ![[E1]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e1"
// CHECK-DAG: ![[E2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "e2"
// CHECK-DAG: ![[E4]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e4"
// CHECK-DAG: ![[S64]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "s64"{{.*}}, size: 64,
// CHECK-DAG: ![[U64]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "u64"{{.*}}, size: 64,
// CHECK-DAG: ![[E5]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e5"
