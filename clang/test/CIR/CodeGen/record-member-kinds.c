// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

struct E {};
// CIR-DAG: !rec_E = !cir.struct<"E" {}>

// In C, containment of an empty record recurses, so ContainsEmpty is itself
// empty for the ABI.  The same declaration in C++ is not.
struct ContainsEmpty { struct E e; };
// CIR-DAG: !rec_ContainsEmpty = !cir.struct<"ContainsEmpty" {empty !rec_E}>

struct ContainsEmptyAndInt { struct E e; int i; };
// CIR-DAG: !rec_ContainsEmptyAndInt = !cir.struct<"ContainsEmptyAndInt" {empty !rec_E, data !s32i}>

struct EmptyArr { struct E e[2]; };
// CIR-DAG: !rec_EmptyArr = !cir.struct<"EmptyArr" {empty !cir.array<!rec_E x 2>}>

// isEmptyFieldForABI peels every array dimension, not just one.
struct MultiDimEmpty { struct E e[2][3]; };
// CIR-DAG: !rec_MultiDimEmpty = !cir.struct<"MultiDimEmpty" {empty !cir.array<!cir.array<!rec_E x 3> x 2>}>

struct ZeroLenArr { int a[0]; };
// CIR-DAG: !rec_ZeroLenArr = !cir.struct<"ZeroLenArr" {empty !cir.array<!s32i x 0>}>

// A flexible array member is data, and keeps its record non-empty.
struct Fam { int n; int a[]; };
// CIR-DAG: !rec_Fam = !cir.struct<"Fam" {data !s32i, data !cir.array<!s32i x 0>}>

// A trailing array of empty records is empty only when its length is constant.
struct FamOfEmpty { struct E e; struct E a[]; };
// CIR-DAG: !rec_FamOfEmpty = !cir.struct<"FamOfEmpty" {empty !rec_E, data !cir.array<!rec_E x 0>}>

struct ZeroLenOfEmpty { struct E e; struct E a[0]; };
// CIR-DAG: !rec_ZeroLenOfEmpty = !cir.struct<"ZeroLenOfEmpty" {empty !rec_E, empty !cir.array<!rec_E x 0>}>

struct UnnamedBitOnly { int : 8; };
// CIR-DAG: !rec_UnnamedBitOnly = !cir.struct<"UnnamedBitOnly" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8, unnamed>]>}>

struct UnnamedBitThenField { int : 8; int f; };
// CIR-DAG: !rec_UnnamedBitThenField = !cir.struct<"UnnamedBitThenField" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8, unnamed>]>, data !s32i}>

// The trailing unit is narrower than its bit-field's declared type and is not
// pad, so it stays in the data size.
struct NamedFieldThenUnnamedBit { char c; int : 24; };
// CIR-DAG: !rec_NamedFieldThenUnnamedBit = !cir.struct<"NamedFieldThenUnnamedBit" {data !s8i, empty !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 24, unnamed>]>}>

// The discrete ms_struct path allocates a unit per formal type.  A unit no
// field of the source names holds no data.
struct MsOnlyUnnamed { int : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsOnlyUnnamed = !cir.struct<"MsOnlyUnnamed" {empty !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>}>

struct MsNamedThenUnnamed { int a : 3; int : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsNamedThenUnnamed = !cir.struct<"MsNamedThenUnnamed" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>, #cir.bitfield_decl<!s32i, 3, unnamed>]>}>

struct MsUnnamedThenNamed { int : 3; int b : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsUnnamedThenNamed = !cir.struct<"MsUnnamedThenNamed" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3, unnamed>, #cir.bitfield_decl<!s32i, 3>]>}>

// A differing formal type starts a new unit, so each of these fields sits in
// one of its own.
struct MsMixed { int a : 3; char : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsMixed = !cir.struct<"MsMixed" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>]>, empty !cir.bitfield<!s8i, [#cir.bitfield_decl<!s8i, 3, unnamed>]>}>

struct MsEmptyFirst { char : 3; int a : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsEmptyFirst = !cir.struct<"MsEmptyFirst" {empty !cir.bitfield<!s8i, [#cir.bitfield_decl<!s8i, 3, unnamed>]>, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>]>}>

struct MsEmptyMiddle {
  int a : 3; char : 3; short b : 3;
} __attribute__((ms_struct));
// CIR-DAG: !rec_MsEmptyMiddle = !cir.struct<"MsEmptyMiddle" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>]>, empty !cir.bitfield<!s8i, [#cir.bitfield_decl<!s8i, 3, unnamed>]>, bitfield !cir.bitfield<!s16i, [#cir.bitfield_decl<!s16i, 3>]>}>

// A zero-width bit-field ends the run here too, so the field after it sits in
// a fresh unit rather than in the one ahead of the split.
struct MsZeroWidthSplit { int a : 3; int : 0; int : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsZeroWidthSplit = !cir.struct<"MsZeroWidthSplit" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>]>, empty !cir.bitfield<[#cir.bitfield_decl<!s32i, 0, unnamed>]>, empty !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>}>

struct MsZeroWidthSplit2 { int : 3; int : 0; int b : 3; } __attribute__((ms_struct));
// CIR-DAG: !rec_MsZeroWidthSplit2 = !cir.struct<"MsZeroWidthSplit2" {empty !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3, unnamed>]>, empty !cir.bitfield<[#cir.bitfield_decl<!s32i, 0, unnamed>]>, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>]>}>

union UnnamedBitUnion { int : 8; };
// CIR-DAG: !rec_UnnamedBitUnion = !cir.union<"UnnamedBitUnion" {empty !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8, unnamed>]>}>

union ContainsEmptyUnion { struct E e; };
// CIR-DAG: !rec_ContainsEmptyUnion = !cir.union<"ContainsEmptyUnion" {empty !rec_E}>

// The two pairs that follow lay out identically, in a struct and in a union
// alike; only the bit-field member's type and mark tell them apart.
struct BitWideUnit { long long x : 32; } __attribute__((aligned(16)));
// CIR-DAG: !rec_BitWideUnit = !cir.struct<"BitWideUnit" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s64i, 32>]>, pad !cir.array<!u8i x 12>}>

struct UIntOverAligned { unsigned x; } __attribute__((aligned(16)));
// CIR-DAG: !rec_UIntOverAligned = !cir.struct<"UIntOverAligned" {data !u32i, pad !cir.array<!u8i x 12>}>

union BitUnit { unsigned a : 1; unsigned b : 1; };
// CIR-DAG: !rec_BitUnit = !cir.union<"BitUnit" {bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!u32i, 1>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!u32i, 1>]>}, padding = {!cir.array<!u8i x 3>}>

union UCharOverAligned { unsigned char c, d; } __attribute__((aligned(4)));
// CIR-DAG: !rec_UCharOverAligned = !cir.union<"UCharOverAligned" {data !u8i, data !u8i}, padding = {!cir.array<!u8i x 3>}>

struct AlignedTail { char c; int i __attribute__((aligned(8))); };
// CIR-DAG: !rec_AlignedTail = !cir.struct<"AlignedTail" {data !s8i, pad !cir.array<!u8i x 7>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.AlignedTail = type { i8, [7 x i8], i32, [4 x i8] }

// Name every record so that its CIR type reaches the output.
void useTypes(struct ContainsEmpty *a, struct ContainsEmptyAndInt *b,
              struct EmptyArr *c, struct MultiDimEmpty *d,
              struct ZeroLenArr *e, struct Fam *f, struct FamOfEmpty *f2,
              struct ZeroLenOfEmpty *f3, struct UnnamedBitOnly *g,
              struct UnnamedBitThenField *h, struct MsOnlyUnnamed *i,
              struct MsNamedThenUnnamed *j, struct MsUnnamedThenNamed *k,
              struct MsMixed *l, struct MsEmptyFirst *m,
              struct MsEmptyMiddle *n, struct MsZeroWidthSplit *o,
              struct MsZeroWidthSplit2 *p, union UnnamedBitUnion *q,
              union ContainsEmptyUnion *r,
              struct NamedFieldThenUnnamedBit *s, struct BitWideUnit *t,
              struct UIntOverAligned *u, union BitUnit *v,
              union UCharOverAligned *w) {}

struct AlignedTail gAlignedTail;

int getAlignedTailI(void) { return gAlignedTail.i; }

// CIR: cir.func{{.*}} @getAlignedTailI()
// CIR:   %[[G:.*]] = cir.get_global @gAlignedTail : !cir.ptr<!rec_AlignedTail>
// CIR:   %{{.*}} = cir.get_member %[[G]][2] {name = "i"} : !cir.ptr<!rec_AlignedTail> -> !cir.ptr<!s32i>
// LLVM: define dso_local i32 @getAlignedTailI()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @gAlignedTail, i64 8), align 8
