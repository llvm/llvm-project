// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,OGCG

struct Empty {};
// CIR-DAG: !rec_Empty = !cir.struct<"Empty" {pad !u8i}>
// LLVM-DAG: %struct.Empty = type { i8 }

// A C++ empty member is not empty for the ABI without [[no_unique_address]].
struct HoldsEmpty { Empty e; int i; };
// CIR-DAG: !rec_HoldsEmpty = !cir.struct<"HoldsEmpty" {data !rec_Empty, data !s32i}>

// A [[no_unique_address]] empty field is elided from a struct's layout, so it
// never becomes a member.  A union keeps its variants, so that is where the
// mark shows.
struct NuaEmpty { [[no_unique_address]] Empty e; int i; };
// CIR-DAG: !rec_NuaEmpty = !cir.struct<"NuaEmpty" {data !s32i}>

union NuaEmptyUnion { [[no_unique_address]] Empty e; int i; };
// CIR-DAG: !rec_NuaEmptyUnion = !cir.union<"NuaEmptyUnion" {empty !rec_Empty, data !s32i}>

// A [[no_unique_address]] field can itself be empty for layout but not for
// the ABI, when its own member is an empty record without the attribute.
// Such a field keeps a real member (and CIR field index) so it is addressed
// via get_member rather than an offset computation.  Classic codegen never
// reifies a distinct member for it at all, so LLVMCIR and OGCG diverge in
// shape (struct-relative GEP vs. raw byte GEP) though not in the byte offset
// they compute -- see the get*/LLVMCIR/OGCG checks near the bottom of this
// file for each of the cases below.
struct EmptyForLayoutOnly { Empty e; };
// CIR-DAG: !rec_EmptyForLayoutOnly = !cir.struct<"EmptyForLayoutOnly" {data !rec_Empty}>
// LLVMCIR-DAG: %struct.EmptyForLayoutOnly = type { %struct.Empty }
// OGCG-DAG: %struct.EmptyForLayoutOnly = type { i8 }
struct NuaHoldsAbiData {
  int x;
  [[no_unique_address]] EmptyForLayoutOnly e;
};
// CIR-DAG: !rec_NuaHoldsAbiData = !cir.struct<"NuaHoldsAbiData" {data !s32i, data !rec_EmptyForLayoutOnly}>
// LLVMCIR-DAG: %struct.NuaHoldsAbiData = type { i32, %struct.EmptyForLayoutOnly }
// OGCG-DAG: %struct.NuaHoldsAbiData = type { i32, [4 x i8] }
struct alignas(8) NuaHoldsAbiDataAligned {
    [[no_unique_address]] EmptyForLayoutOnly e;
    NuaHoldsAbiDataAligned();
};
// CIR-DAG: !rec_NuaHoldsAbiDataAligned = !cir.struct<"NuaHoldsAbiDataAligned" {data !rec_EmptyForLayoutOnly, pad !cir.array<!u8i x 7>}>
// LLVMCIR-DAG: %struct.NuaHoldsAbiDataAligned = type { %struct.EmptyForLayoutOnly, [7 x i8] }
// OGCG-DAG: %struct.NuaHoldsAbiDataAligned = type { [8 x i8] }

// Multiple empty members in the nested type, neither with its own
// [[no_unique_address]]: both stay data members of the nested type.
struct Empty2 {};
// CIR-DAG: !rec_Empty2 = !cir.struct<"Empty2" {pad !u8i}>
// LLVM-DAG: %struct.Empty2 = type { i8 }
struct MultiInner { Empty a; Empty2 b; };
// CIR-DAG: !rec_MultiInner = !cir.struct<"MultiInner" {data !rec_Empty, data !rec_Empty2}>
// LLVMCIR-DAG: %struct.MultiInner = type { %struct.Empty, %struct.Empty2 }
// OGCG-DAG: %struct.MultiInner = type { [2 x i8] }
struct NuaMultiInner {
  int x;
  [[no_unique_address]] MultiInner m;
};
// CIR-DAG: !rec_NuaMultiInner = !cir.struct<"NuaMultiInner" {data !s32i, data !rec_MultiInner}>
// LLVMCIR-DAG: %struct.NuaMultiInner = type { i32, %struct.MultiInner }
// OGCG-DAG: %struct.NuaMultiInner = type { i32, [4 x i8] }

// Mixed: one nested member has its own [[no_unique_address]] (so it is fully
// empty for layout and the ABI, and is elided), the other doesn't (so it
// stays as data, making the wrapping field hold ABI data overall).
struct MixedInner {
  [[no_unique_address]] Empty a;
  Empty2 b;
};
// CIR-DAG: !rec_MixedInner = !cir.struct<"MixedInner" {data !rec_Empty2}>
// LLVMCIR-DAG: %struct.MixedInner = type { %struct.Empty2 }
// OGCG-DAG: %struct.MixedInner = type { i8 }
struct NuaMixedInner {
  int x;
  [[no_unique_address]] MixedInner m;
};
// CIR-DAG: !rec_NuaMixedInner = !cir.struct<"NuaMixedInner" {data !s32i, data !rec_MixedInner}>
// LLVMCIR-DAG: %struct.NuaMixedInner = type { i32, %struct.MixedInner }
// OGCG-DAG: %struct.NuaMixedInner = type { i32, [4 x i8] }

// Two [[no_unique_address]] fields side by side: one fully empty (elided,
// reusing the offset of the preceding field), one holding ABI data (kept as
// a real member).
struct TwoFields {
  int x;
  [[no_unique_address]] Empty allEmpty;
  [[no_unique_address]] MultiInner holds;
};
// CIR-DAG: !rec_TwoFields = !cir.struct<"TwoFields" {data !s32i, data !rec_MultiInner}>
// LLVMCIR-DAG: %struct.TwoFields = type { i32, %struct.MultiInner }
// OGCG-DAG: %struct.TwoFields = type { i32, [4 x i8] }

// Nested two levels deep: the attribute only sits on the outermost field.
struct DeepInner { MultiInner inner; };
// CIR-DAG: !rec_DeepInner = !cir.struct<"DeepInner" {data !rec_MultiInner}>
struct DeepOuter {
  int x;
  [[no_unique_address]] DeepInner d;
};
// CIR-DAG: !rec_DeepOuter = !cir.struct<"DeepOuter" {data !s32i, data !rec_DeepInner}>
// LLVMCIR-DAG: %struct.DeepOuter = type { i32, %struct.DeepInner }
// OGCG-DAG: %struct.DeepOuter = type { i32, [4 x i8] }

// A polymorphic class is never empty for the ABI: its vtable pointer is neither
// a base nor a field.
struct Poly { virtual ~Poly(); };
union NuaPolyUnion { [[no_unique_address]] Poly p; int i; };
// CIR-DAG: !rec_NuaPolyUnion = !cir.union<"NuaPolyUnion" {data !rec_Poly, data !s32i}>

// Emptiness recurses through base classes, in both directions.
struct DerivesEmpty : Empty {};
union NuaDerivedUnion { [[no_unique_address]] DerivesEmpty d; int i; };
// CIR-DAG: !rec_NuaDerivedUnion = !cir.union<"NuaDerivedUnion" {empty !rec_DerivesEmpty, data !s32i}>

struct Pod2 { char c; int i; };
struct DerivesNonEmpty : Pod2 {};
union NuaDerivesNonEmptyUnion { [[no_unique_address]] DerivesNonEmpty d; int i; };
// CIR-DAG: !rec_NuaDerivesNonEmptyUnion = !cir.union<"NuaDerivesNonEmptyUnion" {data !rec_DerivesNonEmpty, data !s32i}>

// The bit-field itself is unnamed, so no field of the source reads it and it
// is marked empty, but it holds data for the ABI.
struct BitFieldBase { int : 3; };
// CIR-DAG: !rec_BitFieldBase = !cir.struct<"BitFieldBase" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>}>
struct DerivesBitFieldBase : BitFieldBase { int i; };
// CIR-DAG: !rec_DerivesBitFieldBase = !cir.struct<"DerivesBitFieldBase" {data !rec_BitFieldBase, data !s32i}>

// A virtual base is marked the same way a non-virtual one is.
struct HasBitFieldVBase : virtual BitFieldBase { int i; };
// CIR-DAG: !rec_HasBitFieldVBase = !cir.struct<"HasBitFieldVBase" packed {data !cir.vptr, data !s32i, data !rec_BitFieldBase, pad !cir.array<!u8i x 3>}>

struct ZeroWidthWide { char c; long long : 0; char d; };
// CIR-DAG: !rec_ZeroWidthWide = !cir.struct<"ZeroWidthWide" {data !s8i, pad !cir.array<!u8i x 7>, empty !cir.bitfield<[#cir.bitfield_decl<!s64i, 0, unnamed>]>, data !s8i}>

struct ZeroLenEmptyArr { Empty e[0]; };
// CIR-DAG: !rec_ZeroLenEmptyArr = !cir.struct<"ZeroLenEmptyArr" {empty !cir.array<!rec_Empty x 0>}>

// A C++ record field is data without the attribute, array or not.
struct EmptyArr2 { Empty e[2]; };
// CIR-DAG: !rec_EmptyArr2 = !cir.struct<"EmptyArr2" {data !cir.array<!rec_Empty x 2>}>

// The [[no_unique_address]] exception covers a record, not an array of them,
// so this field stays in the layout and holds data.
struct NuaEmptyArr { [[no_unique_address]] Empty e[2]; int i; };
// CIR-DAG: !rec_NuaEmptyArr = !cir.struct<"NuaEmptyArr" {data !cir.array<!rec_Empty x 2>, data !s32i}>

struct AlignasTail { char c; alignas(8) int i; };
// CIR-DAG: !rec_AlignasTail = !cir.struct<"AlignasTail" {data !s8i, pad !cir.array<!u8i x 7>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.AlignasTail = type { i8, [7 x i8], i32, [4 x i8] }

// An unnamed bit-field is declared storage that holds no ABI data.  Its member
// carries the type it was declared with alongside the unit it owns.
struct OnlyUnnamedBit { int : 24; };
// CIR-DAG: !rec_OnlyUnnamedBit = !cir.struct<"OnlyUnnamedBit" {empty !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 24, unnamed>]>}>

// A named bit-field holds data whichever order the fields of its unit come in,
// and however the unit is spelled.  The unit is stored as the type the layout
// chose for it, not as any field's declared type.
struct NamedClipped { int i; int j : 24; };
// CIR-DAG: !rec_NamedClipped = !cir.struct<"NamedClipped" {data !s32i, bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s32i, 24>]>}>

struct NamedFirst { int a : 8; int : 16; };
// CIR-DAG: !rec_NamedFirst = !cir.struct<"NamedFirst" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s32i, 8>, #cir.bitfield_decl<!s32i, 16, unnamed>]>}>

struct UnnamedFirst { int : 16; int a : 8; };
// CIR-DAG: !rec_UnnamedFirst = !cir.struct<"UnnamedFirst" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s32i, 16, unnamed>, #cir.bitfield_decl<!s32i, 8>]>}>

// A zero-length bit-field separates one span into two units, so the field
// after it sits in a unit of its own.  A record can carry a named unit and an
// unnamed one at once, in either order.
struct SpanMixed { int a : 3; int : 0; int : 3; };
// CIR-DAG: !rec_SpanMixed = !cir.struct<"SpanMixed" {bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3>]>, pad !cir.array<!u8i x 3>, empty !cir.bitfield<[#cir.bitfield_decl<!s32i, 0, unnamed>]>, empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>, pad !cir.array<!u8i x 3>}>

struct SpanEmptyFirst { int : 3; int : 0; int b : 3; };
// CIR-DAG: !rec_SpanEmptyFirst = !cir.struct<"SpanEmptyFirst" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>, pad !cir.array<!u8i x 3>, empty !cir.bitfield<[#cir.bitfield_decl<!s32i, 0, unnamed>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3>]>, pad !cir.array<!u8i x 3>}>

// One unit covering both of these would need more than one register, so each
// field sits in a unit of its own.
struct WideSpanMixed { unsigned long a : 64; unsigned long : 64; };
// CIR-DAG: !rec_WideSpanMixed = !cir.struct<"WideSpanMixed" {bitfield !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64>]>, empty !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64, unnamed>]>}>

struct WideSpanEmptyFirst { unsigned long : 64; unsigned long b : 64; };
// CIR-DAG: !rec_WideSpanEmptyFirst = !cir.struct<"WideSpanEmptyFirst" {empty !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64, unnamed>]>, bitfield !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64>]>}>

struct WideSpanAllEmpty { unsigned long : 64; unsigned long : 64; };
// CIR-DAG: !rec_WideSpanAllEmpty = !cir.struct<"WideSpanAllEmpty" {empty !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64, unnamed>]>, empty !cir.bitfield<!u64i, [#cir.bitfield_decl<!u64i, 64, unnamed>]>}>

union UnnamedBitUnion { int : 8; };
// CIR-DAG: !rec_UnnamedBitUnion = !cir.union<"UnnamedBitUnion" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8, unnamed>]>}>

union NoMemberUnion {};
// CIR-DAG: !rec_NoMemberUnion = !cir.union<"NoMemberUnion" {}, padding = {!u8i}>

// Natural alignment leaves no gap, so nothing here is pad.
struct Pod { char c; int i; };
// CIR-DAG: !rec_Pod = !cir.struct<"Pod" {data !s8i, data !s32i}>

struct NearlyEmptyVBase { virtual ~NearlyEmptyVBase(); };
// CIR-DAG: !rec_NearlyEmptyVBase = !cir.struct<"NearlyEmptyVBase" {data !cir.vptr}>

struct HasNearlyEmptyVBase : virtual NearlyEmptyVBase { int i; };
// CIR-DAG: !rec_HasNearlyEmptyVBase = !cir.struct<"HasNearlyEmptyVBase" packed {data !rec_NearlyEmptyVBase, data !s32i, pad !cir.array<!u8i x 4>}>

// Only the pad member is reusable, so the unit stays in the base subobject type
// while the byte after it does not.  A named unit stays the same way.
struct Clipped { Clipped(const Clipped &); int i; int : 24; };
// CIR-DAG: !rec_Clipped = !cir.struct<"Clipped" packed {data !s32i, empty !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 24, unnamed>]>, pad !u8i}>
// CIR-DAG: !rec_Clipped2Ebase = !cir.struct<"Clipped.base" packed {data !s32i, empty !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 24, unnamed>]>}>

struct NamedClippedTail { NamedClippedTail(const NamedClippedTail &); int i; int j : 24; };
// CIR-DAG: !rec_NamedClippedTail = !cir.struct<"NamedClippedTail" packed {data !s32i, bitfield !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 24>]>, pad !u8i}>

struct DerivedClipped : Clipped { char c; };
// CIR-DAG: !rec_DerivedClipped = !cir.struct<"DerivedClipped" {data !rec_Clipped2Ebase, data !s8i}>
// LLVM-DAG: %struct.Clipped.base = type <{ i32, [3 x i8] }>
// LLVM-DAG: %struct.DerivedClipped = type { %struct.Clipped.base, i8 }

// Name every record so that its CIR type reaches the output.
void useTypes(HoldsEmpty *, NuaEmpty *, NuaEmptyUnion *, NuaPolyUnion *,
              NuaDerivedUnion *, NuaDerivesNonEmptyUnion *, BitFieldBase *,
              DerivesBitFieldBase *, HasBitFieldVBase *, ZeroWidthWide *,
              ZeroLenEmptyArr *,
              EmptyArr2 *, NuaEmptyArr *, OnlyUnnamedBit *, NamedClipped *,
              NamedFirst *, UnnamedFirst *, SpanMixed *, SpanEmptyFirst *,
              WideSpanMixed *, WideSpanEmptyFirst *, WideSpanAllEmpty *,
              UnnamedBitUnion *, NoMemberUnion *, Pod *, NearlyEmptyVBase *,
              HasNearlyEmptyVBase *, Clipped *, NamedClippedTail *,
              NuaHoldsAbiDataAligned *) {
  EmptyForLayoutOnly e;
  NuaHoldsAbiData nhad;
  NuaHoldsAbiDataAligned nhada;
  Empty2 e2;
  MultiInner mi;
  MixedInner mi2;
  NuaMultiInner nmi;
  NuaMixedInner nmi2;
  TwoFields tf;
  DeepOuter deep;
}

// The zero-length bit-field member leaves the record's alignment alone, so
// these still agree with the classic layout.
SpanMixed gSpanMixed;
// LLVM-DAG: @gSpanMixed = global %struct.SpanMixed zeroinitializer, align 4
SpanEmptyFirst gSpanEmptyFirst;
// LLVM-DAG: @gSpanEmptyFirst = global %struct.SpanEmptyFirst zeroinitializer, align 4
Empty gEmpty;
AlignasTail gAlignasTail;

int getAlignasTailI() { return gAlignasTail.i; }

// CIR: cir.func{{.*}} @_Z15getAlignasTailIv()
// CIR:   %[[G:.*]] = cir.get_global @gAlignasTail : !cir.ptr<!rec_AlignasTail>
// CIR:   %{{.*}} = cir.get_member %[[G]][2] {name = "i"} : !cir.ptr<!rec_AlignasTail> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z15getAlignasTailIv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @gAlignasTail, i64 8), align 8

int getDerivedC(DerivedClipped &d) { return d.c; }

// CIR: cir.func{{.*}} @_Z11getDerivedCR14DerivedClipped(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "c"} : !cir.ptr<!rec_DerivedClipped> -> !cir.ptr<!s8i>
// LLVM: define dso_local noundef i32 @_Z11getDerivedCR14DerivedClipped(ptr noundef nonnull align 4 dereferenceable(8) %{{.+}})
// LLVM:   getelementptr inbounds nuw %struct.DerivedClipped, ptr %{{.+}}, i32 0, i32 1

// A [[no_unique_address]] field that holds ABI data is addressed the same
// way as any other field with a CIR index: get_member, lowering to a
// struct-relative GEP.  Classic codegen never gives the field a distinct
// member, so it instead computes the same byte offset (4, here) with a raw
// i8 GEP against the same base pointer -- the two forms compute identical
// addresses via different means.
EmptyForLayoutOnly *getNuaHoldsAbiDataE(NuaHoldsAbiData &s) { return &s.e; }
// CIR: cir.func{{.*}} @_Z19getNuaHoldsAbiDataE{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "e"} : !cir.ptr<!rec_NuaHoldsAbiData> -> !cir.ptr<!rec_EmptyForLayoutOnly>
// LLVMCIR: define {{.*}} @_Z19getNuaHoldsAbiDataE{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.NuaHoldsAbiData, ptr %{{.+}}, i32 0, i32 1
// OGCG: define {{.*}} @_Z19getNuaHoldsAbiDataE{{.*}}(
// OGCG:   getelementptr inbounds i8, ptr %{{.+}}, i64 4

EmptyForLayoutOnly *getNuaHoldsAbiDataAlignedE(NuaHoldsAbiDataAligned &s) { return &s.e; }
// CIR: cir.func{{.*}} @_Z26getNuaHoldsAbiDataAlignedE{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[0] {name = "e"} : !cir.ptr<!rec_NuaHoldsAbiDataAligned> -> !cir.ptr<!rec_EmptyForLayoutOnly>
// LLVMCIR: define {{.*}} @_Z26getNuaHoldsAbiDataAlignedE{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.NuaHoldsAbiDataAligned, ptr %{{.+}}, i32 0, i32 0
// OGCG: define {{.*}} @_Z26getNuaHoldsAbiDataAlignedE{{.*}}(
// OGCG:   load ptr, ptr

MultiInner *getNuaMultiInnerM(NuaMultiInner &s) { return &s.m; }
// CIR: cir.func{{.*}} @_Z17getNuaMultiInnerM{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "m"} : !cir.ptr<!rec_NuaMultiInner> -> !cir.ptr<!rec_MultiInner>
// LLVMCIR: define {{.*}} @_Z17getNuaMultiInnerM{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.NuaMultiInner, ptr %{{.+}}, i32 0, i32 1
// OGCG: define {{.*}} @_Z17getNuaMultiInnerM{{.*}}(
// OGCG:   getelementptr inbounds i8, ptr %{{.+}}, i64 4

MixedInner *getNuaMixedInnerM(NuaMixedInner &s) { return &s.m; }
// CIR: cir.func{{.*}} @_Z17getNuaMixedInnerM{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "m"} : !cir.ptr<!rec_NuaMixedInner> -> !cir.ptr<!rec_MixedInner>
// LLVMCIR: define {{.*}} @_Z17getNuaMixedInnerM{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.NuaMixedInner, ptr %{{.+}}, i32 0, i32 1
// OGCG: define {{.*}} @_Z17getNuaMixedInnerM{{.*}}(
// OGCG:   getelementptr inbounds i8, ptr %{{.+}}, i64 4

// The fully-empty field reuses offset 0 (no GEP needed at all, in either
// codegen), while the ABI-data-holding field next to it still gets a real
// member / non-zero byte offset.
Empty *getTwoFieldsAllEmpty(TwoFields &s) { return &s.allEmpty; }
// CIR: cir.func{{.*}} @_Z20getTwoFieldsAllEmpty{{.*}}(
// CIR-NOT: cir.get_member
// CIR:   cir.cast bitcast %{{.+}} : !cir.ptr<!rec_TwoFields> -> !cir.ptr<!rec_Empty>
// LLVM: define {{.*}} @_Z20getTwoFieldsAllEmpty{{.*}}(
// LLVM-NOT: getelementptr

MultiInner *getTwoFieldsHolds(TwoFields &s) { return &s.holds; }
// CIR: cir.func{{.*}} @_Z17getTwoFieldsHolds{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "holds"} : !cir.ptr<!rec_TwoFields> -> !cir.ptr<!rec_MultiInner>
// LLVMCIR: define {{.*}} @_Z17getTwoFieldsHolds{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.TwoFields, ptr %{{.+}}, i32 0, i32 1
// OGCG: define {{.*}} @_Z17getTwoFieldsHolds{{.*}}(
// OGCG:   getelementptr inbounds i8, ptr %{{.+}}, i64 4

DeepInner *getDeepOuterD(DeepOuter &s) { return &s.d; }
// CIR: cir.func{{.*}} @_Z13getDeepOuterD{{.*}}(
// CIR:   %{{.*}} = cir.get_member %{{.+}}[1] {name = "d"} : !cir.ptr<!rec_DeepOuter> -> !cir.ptr<!rec_DeepInner>
// LLVMCIR: define {{.*}} @_Z13getDeepOuterD{{.*}}(
// LLVMCIR:   getelementptr inbounds nuw %struct.DeepOuter, ptr %{{.+}}, i32 0, i32 1
// OGCG: define {{.*}} @_Z13getDeepOuterD{{.*}}(
// OGCG:   getelementptr inbounds i8, ptr %{{.+}}, i64 4
