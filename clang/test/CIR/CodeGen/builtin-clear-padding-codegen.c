// RUN: %clang_cc1 -std=c11 -triple=x86_64-linux-gnu -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c11 -triple=x86_64-linux-gnu -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -std=c11 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX,LINUX-OGCG

struct Empty {};

// CIR-LABEL: cir.func no_inline dso_local @testEmpty(
// CIR: %[[ARG:.*]] = cir.alloca "e"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.ptr<!rec_Empty>>, !cir.ptr<!rec_Empty>
// CIR: cir.clear_padding(align(1) %[[LOAD_ARG]], []) : <!rec_Empty> -> ()

// LINUX-LABEL: define dso_local void @testEmpty(
// LINUX-SAME: ptr noundef [[E:%.*]]) #[[ATTR0:[0-9]+]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[E_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[E]], ptr [[E_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[E_ADDR]], align 8
// LINUX-NEXT:    ret void
void testEmpty(struct Empty *e) {
  // Emtpy struct is empty in C in Itanium ABI, no padding
  __builtin_clear_padding(e);
}


// CIR-LABEL: cir.func no_inline dso_local @testPrimitiveNoPadding(
// CIR: %[[I_ARG:.*]] = cir.alloca "i"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[I_ARG]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], []) : <!s32i> -> ()

// LINUX-LABEL: define dso_local void @testPrimitiveNoPadding(
// LINUX-SAME: ptr noundef [[I:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[I_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[I]], ptr [[I_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[I_ADDR]], align 8
// LINUX-NEXT:    ret void

void testPrimitiveNoPadding(int *i) {
  // This should not clear any padding, since int has no padding.
  __builtin_clear_padding(i);
}

// CIR-LABEL: cir.func no_inline dso_local @testPrimitiveLongDouble(
// CIR: %[[LD_ARG:.*]] = cir.alloca "ld"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[LD_ARG]] : !cir.ptr<!cir.ptr<!cir.long_double<!cir.f80>>>, !cir.ptr<!cir.long_double<!cir.f80>>
// CIR: cir.clear_padding(align(16) %[[LOAD_ARG]], [#cir.offset_pair<80, 128>]) : <!cir.long_double<!cir.f80>> -> ()

// LINUX-LABEL: define dso_local void @testPrimitiveLongDouble(
// LINUX-SAME: ptr noundef [[LD:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[LD_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[LD]], ptr [[LD_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[LD_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 10
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 12
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 4
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    ret void
void testPrimitiveLongDouble(long double *ld) {
  // padding [10, 15] on x86
  __builtin_clear_padding(ld);
}

// CIR-LABEL: cir.func no_inline dso_local @testBitInt(
// CIR: %[[BI_ARG:.*]] = cir.alloca "bi"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[BI_ARG]] : !cir.ptr<!cir.ptr<!cir.int<s, 97, bitint>>>, !cir.ptr<!cir.int<s, 97, bitint>>
// CIR: cir.clear_padding(align(8) %[[LOAD_ARG]], [#cir.offset_pair<97, 128>]) : <!cir.int<s, 97, bitint>> -> ()

// LINUX-LABEL: define dso_local void @testBitInt(
// LINUX-SAME: ptr noundef [[BI:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[BI_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[BI]], ptr [[BI_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[BI_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 12
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 4
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], 1
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 4
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    ret void
void testBitInt(_BitInt(97) *bi) {
  // Storage is widened to 128 bits; clear bits [97, 128).
  __builtin_clear_padding(bi);
}

// CIR-LABEL: cir.func no_inline dso_local @testPrimitiveComplexLongDouble(
// CIR: %[[C_ARG:.*]] = cir.alloca "c"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[C_ARG]] : !cir.ptr<!cir.ptr<!cir.complex<!cir.long_double<!cir.f80>>>>, !cir.ptr<!cir.complex<!cir.long_double<!cir.f80>>>
// CIR: cir.clear_padding(align(16) %[[LOAD_ARG]], [#cir.offset_pair<80, 128>, #cir.offset_pair<208, 256>]) : <!cir.complex<!cir.long_double<!cir.f80>>> -> ()

// LINUX-LABEL: define dso_local void @testPrimitiveComplexLongDouble(
// LINUX-SAME: ptr noundef [[C:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[C_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[C]], ptr [[C_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[C_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 10
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 12
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 4
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 26
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 2
// LINUX-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[TMP0]], i32 27
// LINUX-NEXT:    store i8 0, ptr [[TMP8]], align 1
// LINUX-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[TMP0]], i32 28
// LINUX-NEXT:    store i8 0, ptr [[TMP9]], align 4
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[TMP0]], i32 29
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 1
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[TMP0]], i32 30
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 2
// LINUX-NEXT:    [[TMP12:%.*]] = getelementptr i8, ptr [[TMP0]], i32 31
// LINUX-NEXT:    store i8 0, ptr [[TMP12]], align 1
// LINUX-NEXT:    ret void
void testPrimitiveComplexLongDouble(_Complex long double *c) {
  // padding [10, 15] and [26, 31] on x86
  __builtin_clear_padding(c);
}

union U1 {
  int i;
  char c;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnionDifferentLength(
// CIR: %[[U_ARG:.*]] = cir.alloca "u"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[U_ARG]] : !cir.ptr<!cir.ptr<!rec_U1>>, !cir.ptr<!rec_U1>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], []) : <!rec_U1> -> ()

// LINUX-LABEL: define dso_local void @testUnionDifferentLength(
// LINUX-SAME: ptr noundef [[U:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[U_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[U]], ptr [[U_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[U_ADDR]], align 8
// LINUX-NEXT:    ret void
void testUnionDifferentLength(union U1 *u) {
  // This should not clear the object representation bits of the non-active member.
  __builtin_clear_padding(u);
}

struct S {
  __attribute__((aligned(8))) char c1;
};

union U2 {
  struct S s1;
  char c2;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnionTailPaddingOfLongestMember(
// CIR: %[[U_ARG:.*]] = cir.alloca "u"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[U_ARG]] : !cir.ptr<!cir.ptr<!rec_U2>>, !cir.ptr<!rec_U2>
// CIR: cir.clear_padding(align(8) %[[LOAD_ARG]], [#cir.offset_pair<8, 64>]) : <!rec_U2> -> ()

// LINUX-LABEL: define dso_local void @testUnionTailPaddingOfLongestMember(
// LINUX-SAME: ptr noundef [[U:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[U_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[U]], ptr [[U_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[U_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 2
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 2
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 4
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 4
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 6
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 2
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 1
// LINUX-NEXT:    ret void
void testUnionTailPaddingOfLongestMember(union U2 *u) {
  // This should clear the tail padding of the longest member.
  // [1 - 7]
  __builtin_clear_padding(u);
}


struct __attribute__((aligned(4))) Foo {
  char a;
  _Alignas(2) char b;
};

struct __attribute__((aligned(4))) Bar {
  char c;
  _Alignas(2) char d;
};

struct __attribute__((aligned(4))) Baz {
  struct Foo foo;
  char e;
  struct Bar bar;
};

// Baz structure:
// "a", PAD_1, "b", PAD_2, "c", PAD_3, PAD_4, PAD_5, "c", PAD_6, "d", PAD_7
// %struct.Baz = type { %struct.Foo, i8, [3 x i8], %struct.Bar }
// %struct.Foo = type { i8, i8, i8, i8 }
// %struct.Bar = type { i8, i8, i8, i8 }

// CIR-LABEL: cir.func no_inline dso_local @testStructPaddingInBetweenMembers(
// CIR: %[[BAZ_ARG:.*]] = cir.alloca "baz"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[BAZ_ARG]] : !cir.ptr<!cir.ptr<!rec_Baz>>, !cir.ptr<!rec_Baz>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], [#cir.offset_pair<8, 16>, #cir.offset_pair<24, 32>, #cir.offset_pair<40, 64>, #cir.offset_pair<72, 80>, #cir.offset_pair<88, 96>]) : <!rec_Baz> -> ()

// LINUX-LABEL: define dso_local void @testStructPaddingInBetweenMembers(
// LINUX-SAME: ptr noundef [[BAZ:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[BAZ_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[BAZ]], ptr [[BAZ_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[BAZ_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 6
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 2
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 9
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 1
// LINUX-NEXT:    ret void
void testStructPaddingInBetweenMembers(struct Baz *baz) {
  // this should clear all the padding in between various members
  __builtin_clear_padding(baz);
}

// CIR-LABEL: cir.func no_inline dso_local @testStructVolatile(
// CIR: %[[BAZ_ARG:.*]] = cir.alloca "baz"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[BAZ_ARG]] : !cir.ptr<!cir.ptr<!rec_Baz>>, !cir.ptr<!rec_Baz>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], [#cir.offset_pair<8, 16>, #cir.offset_pair<24, 32>, #cir.offset_pair<40, 64>, #cir.offset_pair<72, 80>, #cir.offset_pair<88, 96>]) : <!rec_Baz> -> ()

// LINUX-LABEL: define dso_local void @testStructVolatile(
// LINUX-SAME: ptr noundef [[BAZ:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[BAZ_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[BAZ]], ptr [[BAZ_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[BAZ_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 6
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 2
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 9
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 1
// LINUX-NEXT:    ret void
void testStructVolatile(volatile struct Baz *baz) {
  // this should clear all the padding in between various members
  __builtin_clear_padding(baz);
}




struct S3 {
  long double l;
  _Bool b;
};

// CIR-LABEL: cir.func no_inline dso_local @testStructWithLongDouble(
// CIR: %[[S_ARG:.*]] = cir.alloca "s"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_S3>>, !cir.ptr<!rec_S3>
// CIR: cir.clear_padding(align(16) %[[LOAD_ARG]], [#cir.offset_pair<80, 128>, #cir.offset_pair<136, 256>]) : <!rec_S3> -> ()

// LINUX-LABEL: define dso_local void @testStructWithLongDouble(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 10
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 12
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 4
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 17
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 1
// LINUX-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[TMP0]], i32 18
// LINUX-NEXT:    store i8 0, ptr [[TMP8]], align 2
// LINUX-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[TMP0]], i32 19
// LINUX-NEXT:    store i8 0, ptr [[TMP9]], align 1
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[TMP0]], i32 20
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 4
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[TMP0]], i32 21
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 1
// LINUX-NEXT:    [[TMP12:%.*]] = getelementptr i8, ptr [[TMP0]], i32 22
// LINUX-NEXT:    store i8 0, ptr [[TMP12]], align 2
// LINUX-NEXT:    [[TMP13:%.*]] = getelementptr i8, ptr [[TMP0]], i32 23
// LINUX-NEXT:    store i8 0, ptr [[TMP13]], align 1
// LINUX-NEXT:    [[TMP14:%.*]] = getelementptr i8, ptr [[TMP0]], i32 24
// LINUX-NEXT:    store i8 0, ptr [[TMP14]], align 8
// LINUX-NEXT:    [[TMP15:%.*]] = getelementptr i8, ptr [[TMP0]], i32 25
// LINUX-NEXT:    store i8 0, ptr [[TMP15]], align 1
// LINUX-NEXT:    [[TMP16:%.*]] = getelementptr i8, ptr [[TMP0]], i32 26
// LINUX-NEXT:    store i8 0, ptr [[TMP16]], align 2
// LINUX-NEXT:    [[TMP17:%.*]] = getelementptr i8, ptr [[TMP0]], i32 27
// LINUX-NEXT:    store i8 0, ptr [[TMP17]], align 1
// LINUX-NEXT:    [[TMP18:%.*]] = getelementptr i8, ptr [[TMP0]], i32 28
// LINUX-NEXT:    store i8 0, ptr [[TMP18]], align 4
// LINUX-NEXT:    [[TMP19:%.*]] = getelementptr i8, ptr [[TMP0]], i32 29
// LINUX-NEXT:    store i8 0, ptr [[TMP19]], align 1
// LINUX-NEXT:    [[TMP20:%.*]] = getelementptr i8, ptr [[TMP0]], i32 30
// LINUX-NEXT:    store i8 0, ptr [[TMP20]], align 2
// LINUX-NEXT:    [[TMP21:%.*]] = getelementptr i8, ptr [[TMP0]], i32 31
// LINUX-NEXT:    store i8 0, ptr [[TMP21]], align 1
// LINUX-NEXT:    ret void
void testStructWithLongDouble(struct S3 *s) {
  // "long double data[0-9]", PAD [10-15], "b", PAD [17-31]
  __builtin_clear_padding(s);
}

struct S11 {
  // will usually occupy 2 bytes:
  unsigned char b1 : 3; // 1st 3 bits (in 1st byte) are b1
  unsigned char b2 : 2; // next 2 bits (in 1st byte). The rest bits in byte 1 are unused
  unsigned char b3 : 6; // 6 bits for b3 - doesn't fit into the 1st byte => starts a 2nd
  unsigned char b4 : 2; // 2 bits for b4 - next (and final) bits in the 2nd byte
};

// CIR-LABEL: cir.func no_inline dso_local @testBitFields(
// CIR: %[[S_ARG:.*]] = cir.alloca "s"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_S11>>, !cir.ptr<!rec_S11>
// CIR: cir.clear_padding(align(1) %[[LOAD_ARG]], [#cir.offset_pair<5, 8>]) : <!rec_S11> -> ()

// LINUX-LABEL: define dso_local void @testBitFields(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], 31
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 1
// LINUX-NEXT:    ret void
void testBitFields(struct S11 *s) {
  // "b1" [0-2], "b2" [3-4], PAD [5-7], "b3" [8-13], "b4" [14-15]
  // to clear 5-7, we should AND 0b00011111 (31)
  __builtin_clear_padding(s);
}

// CIR-LABEL: cir.func no_inline dso_local @testArrayNoPadding() attributes {"cir.target-features" = "+cx8,+mmx,+sse,+sse2,+x87", nothrow} {
// CIR: %[[I:.*]] = cir.alloca "i"
// CIR: cir.clear_padding(align(16) %[[I]], []) : <!cir.array<!s32i x 4>> -> ()

// LINUX-LABEL: define dso_local void @testArrayNoPadding(
// LINUX-SAME: ) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[I:%.*]] = alloca [4 x i32], align 16
// LINUX-NEXT:    ret void
void testArrayNoPadding(void) {
  int i[4];
  // there is no padding in the array.
  __builtin_clear_padding(&i);
}

// CIR-LABEL: cir.func no_inline no_proto dso_local @testArrayLongDouble()
// CIR: %[[LD:.*]] = cir.alloca "ld"
// CIR: cir.clear_padding(align(16) %[[LD]], [#cir.offset_pair<80, 128>, #cir.offset_pair<208, 256>]) : <!cir.array<!cir.long_double<!cir.f80> x 2>> -> ()

// LINUX-LABEL: define dso_local void @testArrayLongDouble(
// LINUX-SAME: ) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[LD:%.*]] = alloca [2 x x86_fp80], align 16
// LINUX-NEXT:    [[TMP0:%.*]] = getelementptr i8, ptr [[LD]], i32 10
// LINUX-NEXT:    store i8 0, ptr [[TMP0]], align 2
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[LD]], i32 11
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[LD]], i32 12
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 4
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[LD]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[LD]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 2
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[LD]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[LD]], i32 26
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 2
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[LD]], i32 27
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 1
// LINUX-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[LD]], i32 28
// LINUX-NEXT:    store i8 0, ptr [[TMP8]], align 4
// LINUX-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[LD]], i32 29
// LINUX-NEXT:    store i8 0, ptr [[TMP9]], align 1
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[LD]], i32 30
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 2
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[LD]], i32 31
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 1
// LINUX-NEXT:    ret void
void testArrayLongDouble() {
  // long double 0, [0-9] PAD [10-15]
  // long double 1, [16-25] PAD [26-31]
  long double ld[2];
  __builtin_clear_padding(&ld);
}

// CIR-LABEL: cir.func no_inline dso_local @testArrayOfStruct(
// CIR: %[[S:.*]] = cir.alloca "s"
// CIR: cir.clear_padding(align(16) %[[S]], [#cir.offset_pair<40, 64>, #cir.offset_pair<104, 128>, #cir.offset_pair<168, 192>, #cir.offset_pair<232, 256>]) : <!cir.array<!rec_S_local x 2>> -> ()

// LINUX-LABEL: define dso_local void @testArrayOfStruct(
// LINUX-SAME: ) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S:%.*]] = alloca [2 x [[STRUCT_S_LOCAL:%.*]]], align 16
// LINUX-NEXT:    [[TMP0:%.*]] = getelementptr i8, ptr [[S]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP0]], align 1
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[S]], i32 6
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[S]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[S]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[S]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 2
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[S]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[S]], i32 21
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[S]], i32 22
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 2
// LINUX-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[S]], i32 23
// LINUX-NEXT:    store i8 0, ptr [[TMP8]], align 1
// LINUX-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[S]], i32 29
// LINUX-NEXT:    store i8 0, ptr [[TMP9]], align 1
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[S]], i32 30
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 2
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[S]], i32 31
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 1
// LINUX-NEXT:    ret void
void testArrayOfStruct(void) {
  struct S_local {
    int i1;
    char c1;
    int i2;
    char c2;
  };

  // S[0].i1 [0-3], S[0].c1 [4], PAD [5-7],
  // S[0].i2 [8-11], S[0].c2 [12], PAD [13-15],
  // S[1].i1 [16-19], S[1].c1 [20], PAD [21-23],
  // S[1].i2 [24-27], S[1].c2 [28], PAD [29-31]

  struct S_local s[2];
  __builtin_clear_padding(&s);
}

struct ArrOfStructsWithPadding {
  struct Bar bars[2];
};

// ArrOfStructsWithPadding structure:
// "c" (1), PAD_1, "d" (1), PAD_2, "c" (2), PAD_3, "d" (2), PAD_4
// %struct.ArrOfStructsWithPadding = type { [2 x %struct.Bar] }

// CIR-LABEL: cir.func no_inline dso_local @testArrOfStructsWithPadding(
// CIR: %[[ARR_ARG:.*]] = cir.alloca "arr"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARR_ARG]] : !cir.ptr<!cir.ptr<!rec_ArrOfStructsWithPadding>>, !cir.ptr<!rec_ArrOfStructsWithPadding>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], [#cir.offset_pair<8, 16>, #cir.offset_pair<24, 32>, #cir.offset_pair<40, 48>, #cir.offset_pair<56, 64>]) : <!rec_ArrOfStructsWithPadding> -> ()

// LINUX-LABEL: define dso_local void @testArrOfStructsWithPadding(
// LINUX-SAME: ptr noundef [[ARR:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[ARR_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[ARR]], ptr [[ARR_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARR_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    ret void
void testArrOfStructsWithPadding(struct ArrOfStructsWithPadding *arr) {
  __builtin_clear_padding(arr);
}

// CIR-LABEL: cir.func no_inline dso_local @testAtomic(
// CIR: %[[BAR_ARG:.*]] = cir.alloca "bar"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[BAR_ARG]] : !cir.ptr<!cir.ptr<!rec_Bar>>, !cir.ptr<!rec_Bar>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], [#cir.offset_pair<8, 16>, #cir.offset_pair<24, 32>]) : <!rec_Bar> -> ()

// LINUX-LABEL: define dso_local void @testAtomic(
// LINUX-SAME: ptr noundef [[BAR:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[BAR_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[BAR]], ptr [[BAR_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[BAR_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    ret void
void testAtomic(_Atomic(struct Bar)* bar) {
  __builtin_clear_padding(bar);
}

typedef float Float3Vec __attribute__((ext_vector_type(3)));
typedef long double LongDouble3Vec __attribute__((ext_vector_type(3)));

// CIR-LABEL: cir.func no_inline dso_local @testAttributedType(
// CIR: %[[V_ARG:.*]] = cir.alloca "v"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[V_ARG]] : !cir.ptr<!cir.ptr<!cir.vector<3 x !cir.float>>>, !cir.ptr<!cir.vector<3 x !cir.float>>
// CIR: cir.clear_padding(align(16) %[[LOAD_ARG]], [#cir.offset_pair<96, 128>]) : <!cir.vector<3 x !cir.float>> -> ()

// LINUX-LABEL: define dso_local void @testAttributedType(
// LINUX-SAME: ptr noundef [[V:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[V_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[V]], ptr [[V_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[V_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 12
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 4
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 13
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 14
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 2
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 15
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    ret void
void testAttributedType(Float3Vec* v) {
  __builtin_clear_padding(v);
}

// CIR-LABEL: cir.func no_inline dso_local @testAttributedLongDoubleType(
// CIR: %[[V_ARG:.*]] = cir.alloca "v"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[V_ARG]] : !cir.ptr<!cir.ptr<!cir.vector<3 x !cir.long_double<!cir.f80>>>>, !cir.ptr<!cir.vector<3 x !cir.long_double<!cir.f80>>>
// CIR: cir.clear_padding(align(64) %[[LOAD_ARG]], [#cir.offset_pair<240, 512>]) : <!cir.vector<3 x !cir.long_double<!cir.f80>>> -> ()

// LINUX-LABEL: define dso_local void @testAttributedLongDoubleType(
// LINUX-SAME: ptr noundef [[V:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[V_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[V]], ptr [[V_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[V_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 30
// LINUX-NEXT:    store i8 0, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[TMP0]], i32 31
// LINUX-NEXT:    store i8 0, ptr [[TMP2]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = getelementptr i8, ptr [[TMP0]], i32 32
// LINUX-NEXT:    store i8 0, ptr [[TMP3]], align 32
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 33
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 34
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 35
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 36
// LINUX-NEXT:    store i8 0, ptr [[TMP7]], align 4
// LINUX-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[TMP0]], i32 37
// LINUX-NEXT:    store i8 0, ptr [[TMP8]], align 1
// LINUX-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[TMP0]], i32 38
// LINUX-NEXT:    store i8 0, ptr [[TMP9]], align 2
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[TMP0]], i32 39
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 1
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[TMP0]], i32 40
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 8
// LINUX-NEXT:    [[TMP12:%.*]] = getelementptr i8, ptr [[TMP0]], i32 41
// LINUX-NEXT:    store i8 0, ptr [[TMP12]], align 1
// LINUX-NEXT:    [[TMP13:%.*]] = getelementptr i8, ptr [[TMP0]], i32 42
// LINUX-NEXT:    store i8 0, ptr [[TMP13]], align 2
// LINUX-NEXT:    [[TMP14:%.*]] = getelementptr i8, ptr [[TMP0]], i32 43
// LINUX-NEXT:    store i8 0, ptr [[TMP14]], align 1
// LINUX-NEXT:    [[TMP15:%.*]] = getelementptr i8, ptr [[TMP0]], i32 44
// LINUX-NEXT:    store i8 0, ptr [[TMP15]], align 4
// LINUX-NEXT:    [[TMP16:%.*]] = getelementptr i8, ptr [[TMP0]], i32 45
// LINUX-NEXT:    store i8 0, ptr [[TMP16]], align 1
// LINUX-NEXT:    [[TMP17:%.*]] = getelementptr i8, ptr [[TMP0]], i32 46
// LINUX-NEXT:    store i8 0, ptr [[TMP17]], align 2
// LINUX-NEXT:    [[TMP18:%.*]] = getelementptr i8, ptr [[TMP0]], i32 47
// LINUX-NEXT:    store i8 0, ptr [[TMP18]], align 1
// LINUX-NEXT:    [[TMP19:%.*]] = getelementptr i8, ptr [[TMP0]], i32 48
// LINUX-NEXT:    store i8 0, ptr [[TMP19]], align 16
// LINUX-NEXT:    [[TMP20:%.*]] = getelementptr i8, ptr [[TMP0]], i32 49
// LINUX-NEXT:    store i8 0, ptr [[TMP20]], align 1
// LINUX-NEXT:    [[TMP21:%.*]] = getelementptr i8, ptr [[TMP0]], i32 50
// LINUX-NEXT:    store i8 0, ptr [[TMP21]], align 2
// LINUX-NEXT:    [[TMP22:%.*]] = getelementptr i8, ptr [[TMP0]], i32 51
// LINUX-NEXT:    store i8 0, ptr [[TMP22]], align 1
// LINUX-NEXT:    [[TMP23:%.*]] = getelementptr i8, ptr [[TMP0]], i32 52
// LINUX-NEXT:    store i8 0, ptr [[TMP23]], align 4
// LINUX-NEXT:    [[TMP24:%.*]] = getelementptr i8, ptr [[TMP0]], i32 53
// LINUX-NEXT:    store i8 0, ptr [[TMP24]], align 1
// LINUX-NEXT:    [[TMP25:%.*]] = getelementptr i8, ptr [[TMP0]], i32 54
// LINUX-NEXT:    store i8 0, ptr [[TMP25]], align 2
// LINUX-NEXT:    [[TMP26:%.*]] = getelementptr i8, ptr [[TMP0]], i32 55
// LINUX-NEXT:    store i8 0, ptr [[TMP26]], align 1
// LINUX-NEXT:    [[TMP27:%.*]] = getelementptr i8, ptr [[TMP0]], i32 56
// LINUX-NEXT:    store i8 0, ptr [[TMP27]], align 8
// LINUX-NEXT:    [[TMP28:%.*]] = getelementptr i8, ptr [[TMP0]], i32 57
// LINUX-NEXT:    store i8 0, ptr [[TMP28]], align 1
// LINUX-NEXT:    [[TMP29:%.*]] = getelementptr i8, ptr [[TMP0]], i32 58
// LINUX-NEXT:    store i8 0, ptr [[TMP29]], align 2
// LINUX-NEXT:    [[TMP30:%.*]] = getelementptr i8, ptr [[TMP0]], i32 59
// LINUX-NEXT:    store i8 0, ptr [[TMP30]], align 1
// LINUX-NEXT:    [[TMP31:%.*]] = getelementptr i8, ptr [[TMP0]], i32 60
// LINUX-NEXT:    store i8 0, ptr [[TMP31]], align 4
// LINUX-NEXT:    [[TMP32:%.*]] = getelementptr i8, ptr [[TMP0]], i32 61
// LINUX-NEXT:    store i8 0, ptr [[TMP32]], align 1
// LINUX-NEXT:    [[TMP33:%.*]] = getelementptr i8, ptr [[TMP0]], i32 62
// LINUX-NEXT:    store i8 0, ptr [[TMP33]], align 2
// LINUX-NEXT:    [[TMP34:%.*]] = getelementptr i8, ptr [[TMP0]], i32 63
// LINUX-NEXT:    store i8 0, ptr [[TMP34]], align 1
// LINUX-NEXT:    ret void
void testAttributedLongDoubleType(LongDouble3Vec *v) {
  // long double elements occupy [0-9], [16-25], [32-41] on x86.
  __builtin_clear_padding(v);
}

struct UnnamedBitfieldSingleBit {
  unsigned char a : 3;
  unsigned char   : 1;
  unsigned char b : 4;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnnamedBitfieldSingleBit(
// CIR: %[[S_ARG:.*]] = cir.alloca "s"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldSingleBit>>, !cir.ptr<!rec_UnnamedBitfieldSingleBit>
// CIR: cir.clear_padding(align(1) %[[LOAD_ARG]], [#cir.offset_pair<3, 4>]) : <!rec_UnnamedBitfieldSingleBit> -> ()

// LINUX-LABEL: define dso_local void @testUnnamedBitfieldSingleBit(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], -9
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 1
// LINUX-NEXT:    ret void
void testUnnamedBitfieldSingleBit(struct UnnamedBitfieldSingleBit *s) {
  // byte 0: a[bits 0-2], unnamed[bit 3], b[bits 4-7]
  // bit 3 must be cleared
  // Mask: 0b11110111 == -9
  __builtin_clear_padding(s);
}

struct UnnamedBitfieldMiddle {
  unsigned char a : 3;
  unsigned char   : 2;
  unsigned char b : 3;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnnamedBitfieldMiddle(
// CIR: %[[S_ARG:.*]] = cir.alloca "s"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldMiddle>>, !cir.ptr<!rec_UnnamedBitfieldMiddle>
// CIR: cir.clear_padding(align(1) %[[LOAD_ARG]], [#cir.offset_pair<3, 5>]) : <!rec_UnnamedBitfieldMiddle> -> ()

// LINUX-LABEL: define dso_local void @testUnnamedBitfieldMiddle(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], -25
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 1
// LINUX-NEXT:    ret void
void testUnnamedBitfieldMiddle(struct UnnamedBitfieldMiddle *s) {
  // byte 0: a[0-2], unnamed[3-4], b[5-7]
  // bits 3-4 must be cleared
  // Mask: 0b11100111 == -25
  __builtin_clear_padding(s);
}

struct UnnamedBitfieldSurrounding {
  unsigned char   : 2;
  unsigned char a : 4;
  unsigned char   : 2;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnnamedBitfieldSurrounding(
// CIR: %[[S_ARG:.*]] = cir.alloca "s" align(8) init : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldSurrounding>>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldSurrounding>>, !cir.ptr<!rec_UnnamedBitfieldSurrounding>
// CIR: cir.clear_padding(align(1) %[[LOAD_ARG]], [#cir.offset_pair<0, 2>, #cir.offset_pair<6, 8>]) : <!rec_UnnamedBitfieldSurrounding> -> ()

// LINUX-LABEL: define dso_local void @testUnnamedBitfieldSurrounding(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], -4
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 1
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = and i8 [[TMP5]], 63
// LINUX-NEXT:    store i8 [[TMP6]], ptr [[TMP4]], align 1
// LINUX-NEXT:    ret void
void testUnnamedBitfieldSurrounding(struct UnnamedBitfieldSurrounding *s) {
  // byte 0: unnamed[0-1], a[2-5], unnamed[6-7]
  // bits 0-1 and 6-7 must be cleared
  // Masks:
  //   0b11111100 == -4
  //   0b00111111 == 63
  __builtin_clear_padding(s);
}

struct UnnamedZeroWidthBitfield {
  unsigned char a : 4;
  unsigned int  : 0;
  unsigned int b : 4;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnnamedZeroWidthBitfield(
// CIR: %[[S_ARG:.*]] = cir.alloca "s"
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_UnnamedZeroWidthBitfield>>, !cir.ptr<!rec_UnnamedZeroWidthBitfield>
// CIR: cir.clear_padding(align(4) %[[LOAD_ARG]], [#cir.offset_pair<4, 32>, #cir.offset_pair<36, 64>]) : <!rec_UnnamedZeroWidthBitfield> -> ()

// LINUX-LABEL: define dso_local void @testUnnamedZeroWidthBitfield(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 4
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], 15
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 4
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    store i8 0, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP0]], i32 2
// LINUX-NEXT:    store i8 0, ptr [[TMP5]], align 2
// LINUX-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP0]], i32 3
// LINUX-NEXT:    store i8 0, ptr [[TMP6]], align 1
// LINUX-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP0]], i32 4
// LINUX-NEXT:    [[TMP8:%.*]] = load i8, ptr [[TMP7]], align 4
// LINUX-NEXT:    [[TMP9:%.*]] = and i8 [[TMP8]], 15
// LINUX-NEXT:    store i8 [[TMP9]], ptr [[TMP7]], align 4
// LINUX-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[TMP0]], i32 5
// LINUX-NEXT:    store i8 0, ptr [[TMP10]], align 1
// LINUX-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[TMP0]], i32 6
// LINUX-NEXT:    store i8 0, ptr [[TMP11]], align 2
// LINUX-NEXT:    [[TMP12:%.*]] = getelementptr i8, ptr [[TMP0]], i32 7
// LINUX-NEXT:    store i8 0, ptr [[TMP12]], align 1
// LINUX-NEXT:    ret void
void testUnnamedZeroWidthBitfield(struct UnnamedZeroWidthBitfield *s) {
  // byte 0: a[0-3], unnamed[4-7]
  // bytes 1-3: struct padding
  // bytes 4-7: b[bits 0-3], tail padding[bits 4-31]
  //
  // byte 0:   clear bits 4-7
  //           mask: 0b00001111 == 15
  //
  // byte 1-3: clear all bits
  //           3 x store i8 0
  //
  // byte 4-7: clear bits 4-31
  //           mask: 0b00001111 == 15
  //           3 x store i8 0
  __builtin_clear_padding(s);
}

struct UnnamedBitfieldMultiByte {
  unsigned short a : 4;
  unsigned short   : 8;
  unsigned short b : 4;
};

// CIR-LABEL: cir.func no_inline dso_local @testUnnamedBitfieldMultiByte(
// CIR: %[[S_ARG:.*]] = cir.alloca "s" align(8) init : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldMultiByte>>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[S_ARG]] : !cir.ptr<!cir.ptr<!rec_UnnamedBitfieldMultiByte>>, !cir.ptr<!rec_UnnamedBitfieldMultiByte>
// CIR: cir.clear_padding(align(2) %[[LOAD_ARG]], [#cir.offset_pair<4, 12>]) : <!rec_UnnamedBitfieldMultiByte> -> ()

// LINUX-LABEL: define dso_local void @testUnnamedBitfieldMultiByte(
// LINUX-SAME: ptr noundef [[S:%.*]]) #[[ATTR0]] {
// LINUX-OGCG-NEXT:  [[ENTRY:.*:]]
// LINUX-NEXT:    [[S_ADDR:%.*]] = alloca ptr, align 8
// LINUX-NEXT:    store ptr [[S]], ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[S_ADDR]], align 8
// LINUX-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[TMP0]], i32 0
// LINUX-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP3:%.*]] = and i8 [[TMP2]], 15
// LINUX-NEXT:    store i8 [[TMP3]], ptr [[TMP1]], align 2
// LINUX-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP0]], i32 1
// LINUX-NEXT:    [[TMP5:%.*]] = load i8, ptr [[TMP4]], align 1
// LINUX-NEXT:    [[TMP6:%.*]] = and i8 [[TMP5]], -16
// LINUX-NEXT:    store i8 [[TMP6]], ptr [[TMP4]], align 1
// LINUX-NEXT:    ret void
void testUnnamedBitfieldMultiByte(struct UnnamedBitfieldMultiByte *s) {
  // 2 bytes: a[0-3], unnamed[4-11], b[12-15]
  // byte 0: clear bits 4-7
  // byte 1: clear bits 0-3
  //
  // Masks:
  //   0b00001111 ==  15
  //   0b11110000 == -16
  __builtin_clear_padding(s);
}
