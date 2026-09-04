// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: FileCheck --check-prefix=CIR-NUA --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

struct Base {
  int x;
};

struct Middle : Base {
  char c;
  // sizeof(Middle) = 8 (4 for x, 1 for c, 3 tail padding)
  // data size = 5
};

struct Outer {
  [[no_unique_address]] Middle m;
  char extra;
  Outer(const Middle &m, char e) : m(m), extra(e) {}
};

// The record layout should use the base subobject type for the
// [[no_unique_address]] field, allowing 'extra' to overlap with
// Middle's tail padding.

// CIR: !rec_Middle2Ebase = !cir.struct<"Middle.base" packed {data !rec_Base, data !s8i}>
// CIR: !rec_Outer = !cir.struct<"Outer" {data !rec_Middle2Ebase, data !s8i, pad !cir.array<!u8i x 2>}>

// CIR-LABEL: cir.func {{.*}} @_ZN5OuterC2ERK6Middlec(
// CIR:         %[[THIS:.*]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!rec_Outer>>, !cir.ptr<!rec_Outer>
// CIR:         %[[M_BASE:.*]] = cir.get_member %[[THIS]][0] {name = "m"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_Middle2Ebase>
// CIR-NEXT:    %[[M_COMPLETE:.*]] = cir.cast bitcast %[[M_BASE]] : !cir.ptr<!rec_Middle2Ebase> -> !cir.ptr<!rec_Middle>
// CIR:         cir.copy %{{.+}} align(4) to %[[M_COMPLETE]] align(4) skip_tail_padding : !cir.ptr<!rec_Middle>
// CIR:         %[[EXTRA:.*]] = cir.get_member %[[THIS]][1] {name = "extra"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!s8i>

// Globals for the union/final NUA cases below (placed before LLVM-LABEL so
// these DAG checks anchor to the top of the .ll file rather than to the
// function body).
// LLVM-DAG: %struct.OuterUnion = type { %union.UnionForNUA, i32 }
// LLVM-DAG: %union.UnionForNUA = type { i64 }
// LLVM-DAG: %struct.OuterFinal = type { %struct.FinalForNUA, i8 }
// LLVM-DAG: %struct.FinalForNUA = type { i32, i8 }
// LLVM-DAG: %struct.OuterUnionPad = type { %struct.UnionWithPadding.base, i8 }
// LLVM-DAG: %struct.UnionWithPadding.base = type { i8 }
// LLVM-DAG: %struct.OuterFinalUnionPad = type { %struct.FinalUnionWithPadding.base, i8 }
// LLVM-DAG: %struct.FinalUnionWithPadding.base = type { i8 }
// LLVM-DAG: %struct.OuterUnionPadAfterStorage = type { %struct.UnionPadAfterStorage.base, i8 }
// LLVM-DAG: %struct.UnionPadAfterStorage.base = type <{ i32, [3 x i8] }>
// LLVM-DAG: @oupas = {{(dso_local )?}}global %struct.OuterUnionPadAfterStorage zeroinitializer, align 4
// LLVM-DAG: @ou = {{(dso_local )?}}global %struct.OuterUnion zeroinitializer, align 8
// LLVM-DAG: @of = {{(dso_local )?}}global %struct.OuterFinal zeroinitializer, align 4
// LLVM-DAG: @oup = {{(dso_local )?}}global %struct.OuterUnionPad zeroinitializer, align 2
// LLVM-DAG: @ofup = {{(dso_local )?}}global %struct.OuterFinalUnionPad zeroinitializer, align 2
// LLVM-DAG: %struct.OuterZeroData = type { %union.UnionZeroDataSize, i8 }
// LLVM-DAG: %union.UnionZeroDataSize = type { i32 }
// LLVM-DAG: @ozd = {{(dso_local )?}}global %struct.OuterZeroData zeroinitializer, align 4
// LLVM-DAG: %struct.OuterAllEmpty = type { i8 }
// LLVM-DAG: @oae = {{(dso_local )?}}global %struct.OuterAllEmpty zeroinitializer, align 1
// LLVM-DAG: %struct.OuterUnionBitPad = type { %struct.UnionBitAndWide.base, i8, [6 x i8] }
// LLVM-DAG: %struct.UnionBitAndWide.base = type <{ double, i8 }>
// LLVM-DAG: %struct.OuterAllEmptyBits = type { %struct.UnionAllEmptyBits.base, i8, [4 x i8] }
// LLVM-DAG: %struct.UnionAllEmptyBits.base = type { [3 x i8] }
// LLVM-DAG: @oubp = {{(dso_local )?}}global %struct.OuterUnionBitPad zeroinitializer, align 8
// LLVM-DAG: @oaeb = {{(dso_local )?}}global %struct.OuterAllEmptyBits zeroinitializer, align 8
// OGCG-DAG: %struct.OuterUnion = type { %union.UnionForNUA, i32 }
// OGCG-DAG: %union.UnionForNUA = type { i64 }
// OGCG-DAG: %struct.OuterFinal = type { %struct.FinalForNUA, i8 }
// OGCG-DAG: %struct.FinalForNUA = type { i32, i8 }
// OGCG-DAG: %struct.OuterUnionPad = type { %union.UnionWithPadding.base, i8 }
// OGCG-DAG: %union.UnionWithPadding.base = type { i8 }
// OGCG-DAG: %struct.OuterFinalUnionPad = type { %union.FinalUnionWithPadding.base, i8 }
// OGCG-DAG: %union.FinalUnionWithPadding.base = type { i8 }
// OGCG-DAG: %struct.OuterUnionPadAfterStorage = type { %union.UnionPadAfterStorage.base, i8 }
// OGCG-DAG: %union.UnionPadAfterStorage.base = type <{ i32, [3 x i8] }>
// OGCG-DAG: @oupas = {{(dso_local )?}}global %struct.OuterUnionPadAfterStorage zeroinitializer, align 4
// OGCG-DAG: @ou = {{(dso_local )?}}global %struct.OuterUnion zeroinitializer, align 8
// OGCG-DAG: @of = {{(dso_local )?}}global %struct.OuterFinal zeroinitializer, align 4
// OGCG-DAG: @oup = {{(dso_local )?}}global %struct.OuterUnionPad zeroinitializer, align 2
// OGCG-DAG: @ofup = {{(dso_local )?}}global %struct.OuterFinalUnionPad zeroinitializer, align 2
// OGCG-DAG: %struct.OuterZeroData = type { %union.UnionZeroDataSize, i8 }
// OGCG-DAG: %union.UnionZeroDataSize = type { i32 }
// OGCG-DAG: @ozd = {{(dso_local )?}}global %struct.OuterZeroData zeroinitializer, align 4
// OGCG-DAG: %struct.OuterAllEmpty = type { i8 }
// OGCG-DAG: @oae = {{(dso_local )?}}global %struct.OuterAllEmpty zeroinitializer, align 1
// OGCG-DAG: %struct.OuterUnionBitPad = type { %union.UnionBitAndWide.base, i8, [6 x i8] }
// OGCG-DAG: %union.UnionBitAndWide.base = type <{ double, i8 }>
// OGCG-DAG: %struct.OuterAllEmptyBits = type { %union.UnionAllEmptyBits.base, i8, [4 x i8] }
// OGCG-DAG: %union.UnionAllEmptyBits.base = type { [3 x i8] }
// OGCG-DAG: @oubp = {{(dso_local )?}}global %struct.OuterUnionBitPad zeroinitializer, align 8
// OGCG-DAG: @oaeb = {{(dso_local )?}}global %struct.OuterAllEmptyBits zeroinitializer, align 8

// LLVM-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// LLVM:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[GEP]], ptr align 4 %{{.+}}, i64 5, i1 false)

// OGCG-LABEL: define {{.*}} void @_ZN5OuterC2ERK6Middlec(
// OGCG:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 0
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[GEP]], ptr {{.*}} %{{.+}}, i64 5, i1 false)

void test(const Middle &m) {
  Outer o(m, 'x');
}

// Regression test: a [[no_unique_address]] field whose type is a union (or a
// final class) used to crash CIRRecordLowering with an empty SmallVector::back()
// because computeRecordLayout left the base-subobject type unset for those
// kinds of records, and getStorageType(const CXXRecordDecl *) propagated the
// resulting null mlir::Type into the members vector. We now set baseTy = ty
// for all C++ records, so these layouts succeed.

union UnionForNUA {
  int i;
  long l;
};

struct OuterUnion {
  [[no_unique_address]] UnionForNUA u;
  int x;
};

OuterUnion ou;

struct FinalForNUA final {
  int a;
  char b;
};

struct OuterFinal {
  [[no_unique_address]] FinalForNUA f;
  char tail;
};

OuterFinal of;

// A [[no_unique_address]] union field whose union has reusable tail padding.
struct Padded {
  Padded();

private:
  alignas(2) bool b;
};

union UnionWithPadding {
  UnionWithPadding();
  [[no_unique_address]] Padded p;
  bool flag;
};

struct OuterUnionPad {
  [[no_unique_address]] UnionWithPadding u;
  bool tail;
};

OuterUnionPad oup;

// A final union also gets a base-subobject type for tail-padding reuse.
union FinalUnionWithPadding final {
  FinalUnionWithPadding();
  [[no_unique_address]] Padded p;
  bool flag;
};

struct OuterFinalUnionPad {
  [[no_unique_address]] FinalUnionWithPadding u;
  bool tail;
};

OuterFinalUnionPad ofup;

// A union whose data size outgrows its highest-aligned variant, so the base
// subobject needs padding of its own after the storage type.
struct TailBig {
  TailBig();

private:
  short s;
  char c[5];
};

union UnionPadAfterStorage {
  UnionPadAfterStorage();
  int i;
  [[no_unique_address]] TailBig t;
};

struct OuterUnionPadAfterStorage {
  [[no_unique_address]] UnionPadAfterStorage u;
  bool tail;
};

OuterUnionPadAfterStorage oupas;

// The stand-in member takes a bit-field unit mark as soon as any variant is a
// unit, even where the storage type comes from a variant that is not one.
struct WideTail {
  WideTail();

private:
  double d;
  char c;
};

union UnionBitAndWide {
  UnionBitAndWide();
  [[no_unique_address]] WideTail w;
  unsigned b : 3;
  double d;
};

struct OuterUnionBitPad {
  [[no_unique_address]] UnionBitAndWide u;
  bool tail;
};

OuterUnionBitPad oubp;

// Both variants are unnamed bit-field storage, which holds data for the ABI,
// so the stand-in holds data even though no variant is named.
struct alignas(8) EmptyBitsTail {
  EmptyBitsTail();

private:
  int : 24;
};

union UnionAllEmptyBits {
  UnionAllEmptyBits();
  [[no_unique_address]] EmptyBitsTail e;
  unsigned : 3;
};

struct OuterAllEmptyBits {
  [[no_unique_address]] UnionAllEmptyBits u;
  bool tail;
};

OuterAllEmptyBits oaeb;

// The only variant holding data is a unit, so the stand-in holds data because
// of a unit mark rather than a data mark.
union OnlyBitData {
  OnlyBitData();
  [[no_unique_address]] EmptyBitsTail e;
  unsigned b : 3;
};

struct OuterOnlyBitData {
  [[no_unique_address]] OnlyBitData u;
  bool tail;
};

OuterOnlyBitData oobd;

// CIR-NUA-DAG: !rec_OnlyBitData2Ebase = !cir.struct<"OnlyBitData.base" {bitfield !cir.array<!u8i x 3>}>
// CIR-NUA-DAG: !rec_OuterOnlyBitData = !cir.struct<"OuterOnlyBitData" {data !rec_OnlyBitData2Ebase, data !cir.bool, pad !cir.array<!u8i x 4>}>
// CIR-NUA-DAG: cir.global external @oobd = #cir.zero : !rec_OuterOnlyBitData
// CIR-NUA-DAG: !rec_UnionBitAndWide2Ebase = !cir.struct<"UnionBitAndWide.base" packed {bitfield !cir.double, pad !u8i}>
// CIR-NUA-DAG: !rec_OuterUnionBitPad = !cir.struct<"OuterUnionBitPad" {data !rec_UnionBitAndWide2Ebase, data !cir.bool, pad !cir.array<!u8i x 6>}>
// CIR-NUA-DAG: !rec_UnionAllEmptyBits2Ebase = !cir.struct<"UnionAllEmptyBits.base" {data !cir.array<!u8i x 3>}>
// CIR-NUA-DAG: !rec_OuterAllEmptyBits = !cir.struct<"OuterAllEmptyBits" {data !rec_UnionAllEmptyBits2Ebase, data !cir.bool, pad !cir.array<!u8i x 4>}>
// CIR-NUA-DAG: cir.global external @oubp = #cir.zero : !rec_OuterUnionBitPad
// CIR-NUA-DAG: cir.global external @oaeb = #cir.zero : !rec_OuterAllEmptyBits

// CIR-NUA-DAG: !rec_FinalForNUA = !cir.struct<"FinalForNUA" {data !s32i, data !s8i}>
// CIR-NUA-DAG: !rec_UnionForNUA = !cir.union<"UnionForNUA" {data !s32i, data !s64i}>
// CIR-NUA-DAG: !rec_OuterFinal = !cir.struct<"OuterFinal" {data !rec_FinalForNUA, data !s8i}>
// CIR-NUA-DAG: !rec_OuterUnion = !cir.struct<"OuterUnion" {data !rec_UnionForNUA, data !s32i}>
// CIR-NUA-DAG: !rec_UnionWithPadding2Ebase = !cir.struct<"UnionWithPadding.base" {data !u8i}>
// CIR-NUA-DAG: !rec_OuterUnionPad = !cir.struct<"OuterUnionPad" {data !rec_UnionWithPadding2Ebase, data !cir.bool}>
// CIR-NUA-DAG: !rec_FinalUnionWithPadding2Ebase = !cir.struct<"FinalUnionWithPadding.base" {data !u8i}>
// CIR-NUA-DAG: !rec_OuterFinalUnionPad = !cir.struct<"OuterFinalUnionPad" {data !rec_FinalUnionWithPadding2Ebase, data !cir.bool}>
// CIR-NUA-DAG: !rec_TailBig = !cir.struct<"TailBig" packed {data !s16i, data !cir.array<!s8i x 5>, pad !u8i}>
// CIR-NUA-DAG: !rec_UnionPadAfterStorage2Ebase = !cir.struct<"UnionPadAfterStorage.base" packed {data !s32i, pad !cir.array<!u8i x 3>}>
// CIR-NUA-DAG: !rec_OuterUnionPadAfterStorage = !cir.struct<"OuterUnionPadAfterStorage" {data !rec_UnionPadAfterStorage2Ebase, data !cir.bool}>
// CIR-NUA-DAG: cir.global external @ou = #cir.zero : !rec_OuterUnion
// CIR-NUA-DAG: cir.global external @of = #cir.zero : !rec_OuterFinal
// CIR-NUA-DAG: cir.global external @oup = #cir.zero : !rec_OuterUnionPad
// CIR-NUA-DAG: cir.global external @ofup = #cir.zero : !rec_OuterFinalUnionPad
// CIR-NUA-DAG: cir.global external @oupas = #cir.zero : !rec_OuterUnionPadAfterStorage

struct EmptyForNUA {};

union UnionZeroDataSize {
  [[no_unique_address]] EmptyForNUA e;
  [[no_unique_address]] int i;
};

struct OuterZeroData {
  [[no_unique_address]] UnionZeroDataSize u;
  bool flag;
};

OuterZeroData ozd;

// A union whose members are all [[no_unique_address]] empty types has data
// size 0, but its non-virtual size stays at the 1-byte minimum, so the gate
// does not fire and it needs no distinct base-subobject type.
struct EmptyA {};
struct EmptyB {};

union UnionAllEmpty {
  [[no_unique_address]] EmptyA a;
  [[no_unique_address]] EmptyB b;
};

struct OuterAllEmpty {
  [[no_unique_address]] UnionAllEmpty u;
  bool flag;
};

OuterAllEmpty oae;

// CIR-NUA-DAG: !rec_OuterAllEmpty = !cir.struct<"OuterAllEmpty" {data !cir.bool}>
// CIR-NUA-DAG: cir.global external @oae = #cir.zero : !rec_OuterAllEmpty

// CIR-NUA-DAG: !rec_UnionZeroDataSize = !cir.union<"UnionZeroDataSize" {empty !rec_EmptyForNUA, data !s32i}>
// CIR-NUA-DAG: !rec_OuterZeroData = !cir.struct<"OuterZeroData" {data !rec_UnionZeroDataSize, data !cir.bool}>
// CIR-NUA-DAG: cir.global external @ozd = #cir.zero : !rec_OuterZeroData
