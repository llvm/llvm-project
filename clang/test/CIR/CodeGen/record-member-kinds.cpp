// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

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

// A base holding only unnamed bit-fields is laid out but carries no ABI data,
// which CXXRecordDecl::isEmpty() does not report.
struct BitFieldBase { int : 3; };
// CIR-DAG: !rec_BitFieldBase = !cir.struct<"BitFieldBase" {empty !u8i}>
struct DerivesBitFieldBase : BitFieldBase { int i; };
// CIR-DAG: !rec_DerivesBitFieldBase = !cir.struct<"DerivesBitFieldBase" {empty !rec_BitFieldBase, data !s32i}>

// A virtual base is marked the same way a non-virtual one is.
struct HasBitFieldVBase : virtual BitFieldBase { int i; };
// CIR-DAG: !rec_HasBitFieldVBase = !cir.struct<"HasBitFieldVBase" packed {data !cir.vptr, data !s32i, empty !rec_BitFieldBase, pad !cir.array<!u8i x 3>}>

struct ZeroWidthWide { char c; long long : 0; char d; };
// CIR-DAG: !rec_ZeroWidthWide = !cir.struct<"ZeroWidthWide" {data !s8i, pad !cir.array<!u8i x 7>, bitfield !cir.array<!s64i x 0>, data !s8i}>

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

// An unnamed bit-field unit is declared storage that holds no ABI data.
struct OnlyUnnamedBit { int : 24; };
// CIR-DAG: !rec_OnlyUnnamedBit = !cir.struct<"OnlyUnnamedBit" {empty !cir.array<!u8i x 3>}>

// A unit with a named occupant holds data, whichever order the occupants come
// in, and however the storage is spelled.
struct NamedClipped { int i; int j : 24; };
// CIR-DAG: !rec_NamedClipped = !cir.struct<"NamedClipped" {data !s32i, bitfield !u32i}>

struct NamedFirst { int a : 8; int : 16; };
// CIR-DAG: !rec_NamedFirst = !cir.struct<"NamedFirst" {bitfield !u32i}>

struct UnnamedFirst { int : 16; int a : 8; };
// CIR-DAG: !rec_UnnamedFirst = !cir.struct<"UnnamedFirst" {bitfield !u32i}>

// A zero-length bit-field separates one span into two units and becomes a
// member of its own.  A record can carry a data unit and an empty unit at
// once, in either order.
struct SpanMixed { int a : 3; int : 0; int : 3; };
// CIR-DAG: !rec_SpanMixed = !cir.struct<"SpanMixed" {bitfield !u8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, empty !u8i, pad !cir.array<!u8i x 3>}>

struct SpanEmptyFirst { int : 3; int : 0; int b : 3; };
// CIR-DAG: !rec_SpanEmptyFirst = !cir.struct<"SpanEmptyFirst" {empty !u8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, bitfield !u8i, pad !cir.array<!u8i x 3>}>

// One unit covering both of these would need more than one register, so they
// split into two units that are marked independently.
struct WideSpanMixed { unsigned long a : 64; unsigned long : 64; };
// CIR-DAG: !rec_WideSpanMixed = !cir.struct<"WideSpanMixed" {bitfield !u64i, empty !u64i}>

struct WideSpanEmptyFirst { unsigned long : 64; unsigned long b : 64; };
// CIR-DAG: !rec_WideSpanEmptyFirst = !cir.struct<"WideSpanEmptyFirst" {empty !u64i, bitfield !u64i}>

struct WideSpanAllEmpty { unsigned long : 64; unsigned long : 64; };
// CIR-DAG: !rec_WideSpanAllEmpty = !cir.struct<"WideSpanAllEmpty" {empty !u64i, empty !u64i}>

union UnnamedBitUnion { int : 8; };
// CIR-DAG: !rec_UnnamedBitUnion = !cir.union<"UnnamedBitUnion" {empty !u8i}>

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
// CIR-DAG: !rec_Clipped = !cir.struct<"Clipped" packed {data !s32i, empty !cir.array<!u8i x 3>, pad !u8i}>
// CIR-DAG: !rec_Clipped2Ebase = !cir.struct<"Clipped.base" packed {data !s32i, empty !cir.array<!u8i x 3>}>

struct NamedClippedTail { NamedClippedTail(const NamedClippedTail &); int i; int j : 24; };
// CIR-DAG: !rec_NamedClippedTail = !cir.struct<"NamedClippedTail" packed {data !s32i, bitfield !cir.array<!u8i x 3>, pad !u8i}>

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
              HasNearlyEmptyVBase *, Clipped *, NamedClippedTail *) {}

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
