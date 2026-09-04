// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef struct __attribute__((packed)) { char c; int i; } CharInt;
typedef struct __attribute__((packed)) { char c; int i : 32; } CharIntBF;
typedef struct __attribute__((packed)) { char c; unsigned long long w : 64; } CharWideBF;
typedef struct __attribute__((packed)) { char c; int i : 25; int j : 17; int k : 4; } CharMultipleBFInt;
typedef struct __attribute__((packed)) { char c; int i : 19; } CharUndersizedIntBF;
typedef struct __attribute__((packed)) { char c; int i : 32; char pad[3]; double d; } BFDouble;
typedef struct __attribute__((packed)) { int a; int b; char c; } Nine;
typedef struct __attribute__((packed)) { short a; short b; char c; } FiveShort;
typedef struct __attribute__((packed)) { double d; char c; } DoubleChar;
typedef struct __attribute__((packed)) { float a; float b; char c; } TwoFloatChar;
typedef struct __attribute__((packed)) { int a; int b; char c : 3; } NineBF;
typedef struct __attribute__((packed, aligned(8))) { char c; int i; } PackedOv;
typedef struct __attribute__((packed)) { int a[4]; char c; } Seventeen;
typedef struct { CharInt ci; } NestPacked;
typedef struct { CharInt a[2]; } ArrPacked;
typedef struct { Nine n; } NestNine;
typedef struct { Nine n[1]; } ArrNine;
typedef union __attribute__((packed)) { int i; char c[5]; } UPacked;

#pragma pack(1)
typedef struct { char c; int i; } PragmaPacked;
#pragma pack()

// CIR-DAG: !rec_CharInt = !cir.struct<"CharInt" packed {data !s8i, data !s32i}>
// CIR-DAG: !rec_CharIntBF = !cir.struct<"CharIntBF" packed {data !s8i, bitfield !u32i}>
// CIR-DAG: !rec_CharWideBF = !cir.struct<"CharWideBF" packed {data !s8i, bitfield !u64i}>
// CIR-DAG: !rec_CharMultipleBFInt = !cir.struct<"CharMultipleBFInt" {data !s8i, bitfield !cir.array<!u8i x 6>}>
// CIR-DAG: !rec_CharUndersizedIntBF = !cir.struct<"CharUndersizedIntBF" {data !s8i, bitfield !cir.array<!u8i x 3>}>
// CIR-DAG: !rec_BFDouble = !cir.struct<"BFDouble" packed {data !s8i, bitfield !u32i, data !cir.array<!s8i x 3>, data !cir.double}>
// CIR-DAG: !rec_Nine = !cir.struct<"Nine" packed {data !s32i, data !s32i, data !s8i}>
// CIR-DAG: !rec_FiveShort = !cir.struct<"FiveShort" packed {data !s16i, data !s16i, data !s8i}>
// CIR-DAG: !rec_DoubleChar = !cir.struct<"DoubleChar" packed {data !cir.double, data !s8i}>
// CIR-DAG: !rec_TwoFloatChar = !cir.struct<"TwoFloatChar" packed {data !cir.float, data !cir.float, data !s8i}>
// CIR-DAG: !rec_NineBF = !cir.struct<"NineBF" packed {data !s32i, data !s32i, bitfield !u8i}>
// CIR-DAG: !rec_PackedOv = !cir.struct<"PackedOv" packed {data !s8i, data !s32i, pad !cir.array<!u8i x 3>}>
// CIR-DAG: !rec_Seventeen = !cir.struct<"Seventeen" packed {data !cir.array<!s32i x 4>, data !s8i}>
// CIR-DAG: !rec_NestPacked = !cir.struct<"NestPacked" {data !rec_CharInt}>
// CIR-DAG: !rec_ArrPacked = !cir.struct<"ArrPacked" {data !cir.array<!rec_CharInt x 2>}>
// CIR-DAG: !rec_NestNine = !cir.struct<"NestNine" {data !rec_Nine}>
// CIR-DAG: !rec_ArrNine = !cir.struct<"ArrNine" {data !cir.array<!rec_Nine x 1>}>
// CIR-DAG: !rec_UPacked = !cir.union<"UPacked" packed {data !s32i, data !cir.array<!s8i x 5>}, padding = {!u8i}>
// CIR-DAG: !rec_PragmaPacked = !cir.struct<"PragmaPacked" packed {data !s8i, data !s32i}>

// Anonymous coercion records are numbered in print order, so capture them.
// CIR-DAG: ![[I64I8:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !s8i}>
// CIR-DAG: ![[I64U8:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !u8i}>
// CIR-DAG: ![[F64I8:rec_anon_struct[0-9]*]] = !cir.struct<{data !cir.double, data !s8i}>
// CIR-DAG: ![[V2F32I8:rec_anon_struct[0-9]*]] = !cir.struct<{data !cir.vector<2 x !cir.float>, data !s8i}>

// The int sits at offset 1, and SysV sends a record with a member off its own
// alignment to memory whatever its size.  The same rule takes the return.
int take_char_int(CharInt v) { return v.i; }
CharInt ret_char_int(int x) { CharInt v = {0, x}; return v; }

// CIR: cir.func{{.*}} @take_char_int(%arg0: !cir.ptr<!rec_CharInt> {llvm.align = 8 : i64, llvm.byval = !rec_CharInt, llvm.noundef}{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_char_int(%arg0: !cir.ptr<!rec_CharInt> {llvm.align = 1 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_CharInt, llvm.writable}{{.*}}, %arg1: !s32i {llvm.noundef}{{.*}})
// LLVM: define dso_local i32 @take_char_int(ptr noundef byval(%struct.CharInt) align 8 %{{.+}})
// LLVM: define dso_local void @ret_char_int(ptr dead_on_unwind noalias writable sret(%struct.CharInt) align 1 %{{.+}}, i32 noundef %{{.+}})

// The same record with the int declared as a bit-field.  A bit-field may sit
// at any offset, so the rule above does not reach it and the five bytes stay
// in a register.
int take_char_int_bf(CharIntBF v) { return v.i; }
CharIntBF ret_char_int_bf(int x) { CharIntBF v = {0, x}; return v; }

// A unit that crosses the eightbyte boundary is classified on both sides of it.
int take_char_wide_bf(CharWideBF v) { return (int)v.w; }
CharWideBF ret_char_wide_bf(unsigned long long x) { CharWideBF v = {0, x}; return v; }

// CIR: cir.func{{.*}} @take_char_int_bf(%arg0: !cir.int<u, 40>{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_char_int_bf(%arg0: !s32i {llvm.noundef}{{.*}}) -> !cir.int<u, 40>
// LLVM: define dso_local i32 @take_char_int_bf(i40 %{{.+}})
// LLVM: define dso_local i40 @ret_char_int_bf(i32 noundef %{{.+}})

// CIR: cir.func{{.*}} @take_char_wide_bf(%arg0: !u64i{{.*}}, %arg1: !u8i{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_char_wide_bf(%arg0: !u64i {llvm.noundef}{{.*}}) -> ![[I64U8]]
// LLVM: define dso_local i32 @take_char_wide_bf(i64 %{{.+}}, i8 %{{.+}})
// LLVM: define dso_local { i64, i8 } @ret_char_wide_bf(i64 noundef %{{.+}})

// Three bit-fields share one unit, so the unit spans the bits of all three
// and the record coerces to the seven bytes it occupies.
int take_multiple_bf(CharMultipleBFInt v) { return v.i; }
CharMultipleBFInt ret_multiple_bf(int x) { CharMultipleBFInt v = {0, x, x, x}; return v; }

// CIR: cir.func{{.*}} @take_multiple_bf(%arg0: !cir.int<u, 56>{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_multiple_bf(%arg0: !s32i {llvm.noundef}{{.*}}) -> !cir.int<u, 56>
// LLVM: define dso_local i32 @take_multiple_bf(i56 %{{.+}})
// LLVM: define dso_local i56 @ret_multiple_bf(i32 noundef %{{.+}})

// A unit narrower than the type its bit-field was declared with.  The three
// bytes it occupies and the leading char round up to the same eightbyte.
int take_undersized_bf(CharUndersizedIntBF v) { return v.i; }
CharUndersizedIntBF ret_undersized_bf(int x) { CharUndersizedIntBF v = {0, x}; return v; }

// CIR: cir.func{{.*}} @take_undersized_bf(%arg0: !u32i{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_undersized_bf(%arg0: !s32i {llvm.noundef}{{.*}}) -> !u32i
// LLVM: define dso_local i32 @take_undersized_bf(i32 %{{.+}})
// LLVM: define dso_local i32 @ret_undersized_bf(i32 noundef %{{.+}})

// The unit decides the low eightbyte while the double decides the high one.
double take_bf_double(BFDouble v) { return v.d; }

// CIR: cir.func{{.*}} @take_bf_double(%arg0: !u64i{{.*}}, %arg1: !cir.double{{.*}}) -> !cir.double
// LLVM: define dso_local double @take_bf_double(i64 %{{.+}}, double %{{.+}})

// Every member is naturally aligned and only the nine-byte size earns the
// packed mark, so this one is classified: an eightbyte of ints and a trailing
// byte.
int take_nine(Nine v) { return v.b; }
Nine ret_nine(int x) { Nine v = {0, x, 0}; return v; }

// CIR: cir.func{{.*}} @take_nine(%arg0: !u64i{{.*}}, %arg1: !s8i{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_nine(%arg0: !s32i {llvm.noundef}{{.*}}) -> ![[I64I8]]
// LLVM: define dso_local i32 @take_nine(i64 %{{.+}}, i8 %{{.+}})
// LLVM: define dso_local { i64, i8 } @ret_nine(i32 noundef %{{.+}})

// Same rule inside one eightbyte, so the five bytes coerce to an i40 rather
// than being rounded up to an i64.
int take_five_short(FiveShort v) { return v.b; }
FiveShort ret_five_short(short x) { FiveShort v = {0, x, 0}; return v; }

// CIR: cir.func{{.*}} @take_five_short(%arg0: !cir.int<u, 40>{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_five_short(%arg0: !s16i {llvm.noundef, llvm.signext}{{.*}}) -> !cir.int<u, 40>
// LLVM: define dso_local i32 @take_five_short(i40 %{{.+}})
// LLVM: define dso_local i40 @ret_five_short(i16 noundef signext %{{.+}})

// A named access unit is only ambiguous where padding lets classic read the
// declared type through the gap, so a packed record whose every byte holds
// data is classified rather than refused by the unit-width rule.
int take_ninebf(NineBF v) { return v.b; }

// CIR: cir.func{{.*}} @take_ninebf(%arg0: !u64i{{.*}}, %arg1: !u8i{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_ninebf(i64 %{{.+}}, i8 %{{.+}})

// The two eightbytes land in different register classes.
double take_double_char(DoubleChar v) { return v.d; }
DoubleChar ret_double_char(double x) { DoubleChar v = {x, 0}; return v; }

// CIR: cir.func{{.*}} @take_double_char(%arg0: !cir.double{{.*}}, %arg1: !s8i{{.*}}) -> !cir.double
// CIR: cir.func{{.*}} @ret_double_char(%arg0: !cir.double {llvm.noundef}{{.*}}) -> ![[F64I8]]
// LLVM: define dso_local double @take_double_char(double %{{.+}}, i8 %{{.+}})
// LLVM: define dso_local { double, i8 } @ret_double_char(double noundef %{{.+}})

// Two floats share the low eightbyte, so the SSE rules pack them into a vector
// instead of widening to a double.
float take_two_float_char(TwoFloatChar v) { return v.b; }
TwoFloatChar ret_two_float_char(float x) { TwoFloatChar v = {x, x, 0}; return v; }

// CIR: cir.func{{.*}} @take_two_float_char(%arg0: !cir.vector<2 x !cir.float>{{.*}}, %arg1: !s8i{{.*}}) -> !cir.float
// CIR: cir.func{{.*}} @ret_two_float_char(%arg0: !cir.float {llvm.noundef}{{.*}}) -> ![[V2F32I8]]
// LLVM: define dso_local float @take_two_float_char(<2 x float> %{{.+}}, i8 %{{.+}})
// LLVM: define dso_local { <2 x float>, i8 } @ret_two_float_char(float noundef %{{.+}})

// The other route to memory: every member is aligned, so it is the size past
// two eightbytes that decides.
int take_seventeen(Seventeen v) { return v.a[3]; }

// CIR: cir.func{{.*}} @take_seventeen(%arg0: !cir.ptr<!rec_Seventeen> {llvm.align = 8 : i64, llvm.byval = !rec_Seventeen, llvm.noundef}{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_seventeen(ptr noundef byval(%struct.Seventeen) align 8 %{{.+}})

// Packed and over-aligned at once, so the record carries a pad member and the
// packed mark together.  The misaligned int still decides it.
int take_packed_ov(PackedOv v) { return v.i; }

// CIR: cir.func{{.*}} @take_packed_ov(%arg0: !cir.ptr<!rec_PackedOv> {llvm.align = 8 : i64, llvm.byval = !rec_PackedOv, llvm.noundef}{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_packed_ov(ptr noundef byval(%struct.PackedOv) align 8 %{{.+}})

// A packed member reaches the classifier through an enclosing record and
// through an array element, neither of which is packed itself.  The member
// decides the outcome, so both routes are covered on each side of it.
int take_nest_packed(NestPacked v) { return v.ci.i; }
int take_arr_packed(ArrPacked v) { return v.a[1].i; }
int take_nest_nine(NestNine v) { return v.n.b; }
int take_arr_nine(ArrNine v) { return v.n[0].b; }

// CIR: cir.func{{.*}} @take_nest_packed(%arg0: !cir.ptr<!rec_NestPacked> {llvm.align = 8 : i64, llvm.byval = !rec_NestPacked, llvm.noundef}{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @take_arr_packed(%arg0: !cir.ptr<!rec_ArrPacked> {llvm.align = 8 : i64, llvm.byval = !rec_ArrPacked, llvm.noundef}{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @take_nest_nine(%arg0: !u64i{{.*}}, %arg1: !s8i{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @take_arr_nine(%arg0: !u64i{{.*}}, %arg1: !s8i{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_nest_packed(ptr noundef byval(%struct.NestPacked) align 8 %{{.+}})
// LLVM: define dso_local i32 @take_arr_packed(ptr noundef byval(%struct.ArrPacked) align 8 %{{.+}})
// LLVM: define dso_local i32 @take_nest_nine(i64 %{{.+}}, i8 %{{.+}})
// LLVM: define dso_local i32 @take_arr_nine(i64 %{{.+}}, i8 %{{.+}})

// A union's members all start at offset zero, so packing never misaligns one
// and the five-byte union stays in a register.
int take_upacked(UPacked v) { return v.i; }

// CIR: cir.func{{.*}} @take_upacked(%arg0: !cir.int<u, 40>{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_upacked(i40 %{{.+}})

// #pragma pack reaches the same layout as the attribute.
int take_pragma_packed(PragmaPacked v) { return v.i; }

// CIR: cir.func{{.*}} @take_pragma_packed(%arg0: !cir.ptr<!rec_PragmaPacked> {llvm.align = 8 : i64, llvm.byval = !rec_PragmaPacked, llvm.noundef}{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_pragma_packed(ptr noundef byval(%struct.PragmaPacked) align 8 %{{.+}})
