// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

typedef struct { int x; unsigned : 0; } __attribute__((aligned(16))) Tail;
typedef struct { char c; int : 0; int y; } Mid;
typedef struct { int x; int : 0; int y; } Plain;
typedef struct { char c; long long : 0; } __attribute__((aligned(16))) Wide;
typedef struct { char c; int : 0; short s; int : 0; } Twice;
typedef struct { char c; int : 0; long long : 0; short s; } Adjacent;
typedef struct { int : 0; int x; } __attribute__((aligned(16))) AtStart;
typedef struct { int : 0; } OnlyZeroWidth;
typedef struct { int x; } __attribute__((aligned(16))) NoZeroWidth;
typedef struct { Tail t; } Nest;
typedef struct { Tail t[1]; } ArrayOfTail;
struct { int x; unsigned : 0; } __attribute__((aligned(16))) gAnon;
typedef struct { long long a; int b; unsigned : 0; } Pair;
typedef struct { char c; int : 0; char d[24]; } TooBig;
typedef struct { char c; int : 0; long long y; } AfterZeroWidth;
typedef struct { char c; long long y; } NoZeroWidthPair;
typedef struct { double d; int : 32; } __attribute__((aligned(16))) UnnamedUnit;
typedef struct { char c; long long : 0; char d; } WiderThanRecord;
typedef struct { int a; int : 0; int fam[]; } ZeroWidthAndFam;
typedef struct { char c; int : 0; int y; } __attribute__((packed)) Packed;
typedef struct { int x; unsigned : 0; }
    __attribute__((ms_struct, aligned(16))) MsTail;
typedef union { int a; Mid m; } ZeroWidthUnion;

// An initializer the lowering builds a member at a time writes the empty field
// in place of the bit-field's element, since the field holds no bytes and is
// not the declared type the element carries.  A relocatable and a `_BitInt`
// both take that path, reached through an array element and a member record
// as well.
extern int arr[4];
typedef struct { int *p; int : 0; int y; } InitPtr;
InitPtr gInitPtr = {&arr[2], 5};
typedef struct { _BitInt(7) b; int : 0; int y; } InitBitInt;
InitBitInt gInitBitInt = {3, 4};
InitPtr gInitArr[2] = {{&arr[1], 1}, {&arr[2], 2}};
typedef struct { InitPtr inner; int z; } InitNest;
InitNest gInitNest = {{&arr[3], 7}, 9};
// LLVM-CIR-DAG: @gInitPtr = global %struct.InitPtr { ptr getelementptr{{.*}}(i8, ptr @arr, i64 8), [0 x i8] zeroinitializer, i32 5 }, align 8
// LLVM-OGCG-DAG: @gInitPtr = global { ptr, i32, [4 x i8] } { ptr getelementptr{{.*}}(i8, ptr @arr, i64 8), i32 5, [4 x i8] zeroinitializer }, align 8
// LLVM-CIR-DAG: @gInitBitInt = global %struct.InitBitInt { i8 3, [3 x i8] zeroinitializer, [0 x i8] zeroinitializer, i32 4 }, align 4
// LLVM-OGCG-DAG: @gInitBitInt = global { i8, [3 x i8], i32 } { i8 3, [3 x i8] zeroinitializer, i32 4 }, align 4
// LLVM-CIR-DAG: @gInitArr = global [2 x %struct.InitPtr] [%struct.InitPtr { ptr getelementptr{{.*}}(i8, ptr @arr, i64 4), [0 x i8] zeroinitializer, i32 1 }, %struct.InitPtr { ptr getelementptr{{.*}}(i8, ptr @arr, i64 8), [0 x i8] zeroinitializer, i32 2 }], align 16
// LLVM-CIR-DAG: @gInitNest = global %struct.InitNest { %struct.InitPtr { ptr getelementptr{{.*}}(i8, ptr @arr, i64 12), [0 x i8] zeroinitializer, i32 7 }, i32 9 }, align 8

// A flexible array member's field holds no bytes too, but it does have the
// element's type, so it keeps its own initializer.
typedef struct { int *p; int fam[]; } FamRel;
FamRel gFamRel = {&arr[2]};
// LLVM-DAG: @gFamRel = global %struct.FamRel { ptr getelementptr{{.*}}(i8, ptr @arr, i64 8), [0 x i32] zeroinitializer }, align 8

// CIR-DAG: !rec_Tail = !cir.struct<"Tail" {data !s32i, bitfield !cir.array<!u32i x 0>, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_Mid = !cir.struct<"Mid" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, data !s32i}>
// CIR-DAG: !rec_Plain = !cir.struct<"Plain" {data !s32i, bitfield !cir.array<!s32i x 0>, data !s32i}>
// CIR-DAG: !rec_Wide = !cir.struct<"Wide" {data !s8i, pad !cir.array<!u8i x 7>, bitfield !cir.array<!s64i x 0>, pad !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_Twice = !cir.struct<"Twice" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, data !s16i, pad !cir.array<!u8i x 2>, bitfield !cir.array<!s32i x 0>}>
// CIR-DAG: !rec_Adjacent = !cir.struct<"Adjacent" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, pad !cir.array<!u8i x 4>, bitfield !cir.array<!s64i x 0>, data !s16i}>
// CIR-DAG: !rec_AtStart = !cir.struct<"AtStart" {bitfield !cir.array<!s32i x 0>, data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_Pair = !cir.struct<"Pair" {data !s64i, data !s32i, bitfield !cir.array<!u32i x 0>}>
// CIR-DAG: !rec_TooBig = !cir.struct<"TooBig" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, data !cir.array<!s8i x 24>}>
// CIR-DAG: !rec_AfterZeroWidth = !cir.struct<"AfterZeroWidth" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, data !s64i}>
// CIR-DAG: !rec_WiderThanRecord = !cir.struct<"WiderThanRecord" {data !s8i, pad !cir.array<!u8i x 7>, bitfield !cir.array<!s64i x 0>, data !s8i}>
// CIR-DAG: !rec_Packed = !cir.struct<"Packed" {data !s8i, pad !cir.array<!u8i x 3>, bitfield !cir.array<!s32i x 0>, data !s32i}>

// The Microsoft layout builds bit-field runs on its own path.
// CIR-DAG: !rec_MsTail = !cir.struct<"MsTail" {data !s32i, bitfield !cir.array<!u32i x 0>, pad !cir.array<!u8i x 12>}>

// CIR-DAG: !rec_OnlyZeroWidth = !cir.struct<"OnlyZeroWidth" {bitfield !cir.array<!s32i x 0>}>

// An anonymous record reaches the same path under its generated name.
// CIR-DAG: !rec_anon2E0 = !cir.struct<"anon.0" {data !s32i, bitfield !cir.array<!u32i x 0>, pad !cir.array<!u8i x 12>}>

// A zero-length array under `data` is a flexible array member, which keeps
// its alignment contribution.
// CIR-DAG: !rec_ZeroWidthAndFam = !cir.struct<"ZeroWidthAndFam" {data !s32i, bitfield !cir.array<!s32i x 0>, data !cir.array<!s32i x 0>}>

// CIR-DAG: !rec_NoZeroWidth = !cir.struct<"NoZeroWidth" {data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_NoZeroWidthPair = !cir.struct<"NoZeroWidthPair" {data !s8i, data !s64i}>
// CIR-DAG: !rec_Nest = !cir.struct<"Nest" {data !rec_Tail}>
// CIR-DAG: !rec_ArrayOfTail = !cir.struct<"ArrayOfTail" {data !cir.array<!rec_Tail x 1>}>

// The alias number depends on emission order, so bind it by content.
// CIR-DAG: ![[PAIR_RET:rec_anon_struct[0-9]*]] = !cir.struct<{data !s64i, data !u64i}>

// The lowered member carries no type.  The declared type would make
// `WiderThanRecord` eight-byte aligned where the source says one.
// LLVM-CIR-DAG: %struct.Tail = type { i32, [0 x i8], [12 x i8] }
// LLVM-OGCG-DAG: %struct.Tail = type { i32, [12 x i8] }
// LLVM-CIR-DAG: %struct.Mid = type { i8, [3 x i8], [0 x i8], i32 }
// LLVM-OGCG-DAG: %struct.Mid = type { i8, i32 }
// LLVM-CIR-DAG: %struct.Wide = type { i8, [7 x i8], [0 x i8], [8 x i8] }
// LLVM-OGCG-DAG: %struct.Wide = type { i8, [15 x i8] }
// LLVM-CIR-DAG: %struct.WiderThanRecord = type { i8, [7 x i8], [0 x i8], i8 }
// LLVM-OGCG-DAG: %struct.WiderThanRecord = type { i8, [7 x i8], i8 }

// The bit-field extends the eightbyte's user data past `x`, so it stays i64
// instead of narrowing to i32 the way the same over-aligned shape does without
// one.  Compare take_no_zero_width below.
void take_tail(Tail t) {}
// CIR: cir.func{{.*}} @take_tail(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_tail(i64 %{{.+}})

// `y` already carries the eightbyte's user data to bit 64, so the bit-field
// changes nothing here and in take_plain.
void take_mid(Mid m) {}
// CIR: cir.func{{.*}} @take_mid(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_mid(i64 %{{.+}})

void take_plain(Plain p) {}
// CIR: cir.func{{.*}} @take_plain(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_plain(i64 %{{.+}})

// The bit-field sits at bit 64, past the eightbyte holding `c`, so it widens
// nothing.
void take_wide(Wide w) {}
// CIR: cir.func{{.*}} @take_wide(%arg0: !s8i loc
// LLVM: define{{.*}} void @take_wide(i8 %{{.+}})

void take_twice(Twice t) {}
// CIR: cir.func{{.*}} @take_twice(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_twice(i64 %{{.+}})

void take_adjacent(Adjacent a) {}
// CIR: cir.func{{.*}} @take_adjacent(%arg0: !u64i loc{{.*}}, %arg1: !s16i loc
// LLVM: define{{.*}} void @take_adjacent(i64 %{{.+}}, i16 %{{.+}})

// At offset zero the bit-field reaches no further than `x` does.
void take_at_start(AtStart a) {}
// CIR: cir.func{{.*}} @take_at_start(%arg0: !s32i loc
// LLVM: define{{.*}} void @take_at_start(i32 %{{.+}})

void take_only_zero_width(OnlyZeroWidth o) {}
// CIR: cir.func{{.*}} @take_only_zero_width()
// LLVM: define{{.*}} void @take_only_zero_width()

void take_no_zero_width(NoZeroWidth n) {}
// CIR: cir.func{{.*}} @take_no_zero_width(%arg0: !s32i loc
// LLVM: define{{.*}} void @take_no_zero_width(i32 %{{.+}})

void take_nest(Nest n) {}
// CIR: cir.func{{.*}} @take_nest(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_nest(i64 %{{.+}})

void take_array_of_tail(ArrayOfTail a) {}
// CIR: cir.func{{.*}} @take_array_of_tail(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_array_of_tail(i64 %{{.+}})

void take_anon(__typeof__(gAnon) a) {}
// CIR: cir.func{{.*}} @take_anon(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_anon(i64 %{{.+}})

// The bit-field lands in the second eightbyte, which is classified and coerced
// on its own.
void take_pair(Pair p) {}
// CIR: cir.func{{.*}} @take_pair(%arg0: !s64i loc{{.*}}, %arg1: !u64i loc
// LLVM: define{{.*}} void @take_pair(i64 %{{.+}}, i64 %{{.+}})

void take_after_zero_width(AfterZeroWidth a) {}
// CIR: cir.func{{.*}} @take_after_zero_width(%arg0: !u64i loc{{.*}}, %arg1: !s64i loc
// LLVM: define{{.*}} void @take_after_zero_width(i64 %{{.+}}, i64 %{{.+}})

void take_no_zero_width_pair(NoZeroWidthPair n) {}
// CIR: cir.func{{.*}} @take_no_zero_width_pair(%arg0: !s8i loc{{.*}}, %arg1: !s64i loc
// LLVM: define{{.*}} void @take_no_zero_width_pair(i8 %{{.+}}, i64 %{{.+}})

// The unit occupies four bytes after `d`.  Counting it as data would stop this
// being a single SSE value and pass it as a double plus an i32.
void take_unnamed_unit(UnnamedUnit u) {}
// CIR: cir.func{{.*}} @take_unnamed_unit(%arg0: !cir.double loc
// LLVM: define{{.*}} void @take_unnamed_unit(double %{{.+}})

void take_wider_than_record(WiderThanRecord w) {}
// CIR: cir.func{{.*}} @take_wider_than_record(%arg0: !u64i loc{{.*}}, %arg1: !s8i loc
// LLVM: define{{.*}} void @take_wider_than_record(i64 %{{.+}}, i8 %{{.+}})

void take_zero_width_and_fam(ZeroWidthAndFam *p) {}
// CIR: cir.func{{.*}} @take_zero_width_and_fam(%arg0: !cir.ptr<!rec_ZeroWidthAndFam>
// LLVM: define{{.*}} void @take_zero_width_and_fam(ptr noundef %{{.+}})

void take_packed(Packed p) {}
// CIR: cir.func{{.*}} @take_packed(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_packed(i64 %{{.+}})

void take_ms_tail(MsTail t) {}
// CIR: cir.func{{.*}} @take_ms_tail(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_ms_tail(i64 %{{.+}})

// A union reduces to its largest member, which drops the bit-field with it.
void take_union(ZeroWidthUnion u) {}
// CIR: cir.func{{.*}} @take_union(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_union(i64 %{{.+}})

// Past two eightbytes the record goes to memory, where the widened coerce type
// no longer applies.
void take_too_big(TooBig b) {}
// CIR: cir.func{{.*}} @take_too_big(%arg0: !cir.ptr<!rec_TooBig> {llvm.align = 8 : i64, llvm.byval = !rec_TooBig, llvm.noalias, llvm.noundef}
// LLVM-CIR: define{{.*}} void @take_too_big(ptr noalias noundef byval(%struct.TooBig) align 8 %{{.+}})
// LLVM-OGCG: define{{.*}} void @take_too_big(ptr noundef byval(%struct.TooBig) align 8 %{{.+}})

Tail ret_tail(void) { Tail t = {3}; return t; }
// CIR: cir.func{{.*}} @ret_tail() -> !u64i
// LLVM: define{{.*}} i64 @ret_tail()

Pair ret_pair(void) { Pair p = {1, 2}; return p; }
// CIR: cir.func{{.*}} @ret_pair() -> ![[PAIR_RET]]
// LLVM: define{{.*}} { i64, i64 } @ret_pair()
