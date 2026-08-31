// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

struct Bits {
  unsigned x : 3;
};

struct Inner {
  unsigned x : 3;
};

struct OuterNest {
  Inner i;
};

struct WideInner {
  long long b : 1;
};

struct ArrNest {
  WideInner a[1];
} __attribute__((aligned(16)));

struct Nested {
  unsigned w : 16;
  unsigned l : 13;
};

struct Wrap {
  int a;
  Nested n;
} __attribute__((aligned(16)));

struct UnnamedPastUnit {
  int a : 8;
  int : 16;
} __attribute__((aligned(8)));

struct UnnamedWide {
  unsigned char a : 3;
  unsigned long long : 5;
} __attribute__((aligned(8)));

struct NarrowUnit {
  unsigned x : 32;
} __attribute__((aligned(16)));

struct WideUnit {
  long long x : 32;
} __attribute__((aligned(16)));

struct NamedZeroWidth {
  unsigned x : 3;
  unsigned : 0;
  int y;
};

struct __attribute__((packed, aligned(8))) PackedBits {
  unsigned b : 3;
  int x;
};

struct SpillsPastUnit {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e;
} __attribute__((aligned(8)));

struct UnnamedOnly {
  int : 8;
};

union UUnnamedOnly {
  int : 8;
};

struct Sema {
  unsigned w : 16;
  unsigned l : 13;
  bool s : 1;
};

struct Holder {
  long long v;
  Sema sema;
};

union UBits {
  unsigned a : 1;
  unsigned b : 1;
};

union UBytes {
  unsigned char c, d;
};

union UWideBits {
  unsigned long long w : 3;
  char pad[4];
};

// A unit narrower than the type declared in it carries the declared reach as a
// zero-length member, which is what widens the coercion from the unit's own i8
// to the eightbyte the declaration covers.
// CIR-DAG: !rec_Bits = !cir.struct<"Bits" {bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 3> x 0>, pad !cir.array<!u8i x 3>}>
// CIR-DAG: !rec_Sema = !cir.struct<"Sema" {bitfield !u32i, bitfield !cir.array<!cir.array<!u8i x 2> x 0>}>
// CIR-DAG: !rec_UBits = !cir.union<"UBits" {bitfield !u8i, bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 4> x 0>}, padding = {!cir.array<!u8i x 3>}>
// CIR-DAG: !rec_UWideBits = !cir.union<"UWideBits" {bitfield !u8i, data !cir.array<!s8i x 4>, bitfield !cir.array<!cir.array<!u8i x 8> x 0>}, padding = {!cir.array<!u8i x 4>}>
// CIR-DAG: !rec_WideUnit = !cir.struct<"WideUnit" {bitfield !u32i, bitfield !cir.array<!cir.array<!u8i x 4> x 0>, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_PackedBits = !cir.struct<"PackedBits" packed {bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 3> x 0>, data !s32i, pad !cir.array<!u8i x 3>}>

// A reach and a zero-width bit-field sit in one record without conflating.
// CIR-DAG: !rec_NamedZeroWidth = !cir.struct<"NamedZeroWidth" {bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 3> x 0>, pad !cir.array<!u8i x 3>, bitfield !cir.array<!u32i x 0>, data !s32i}>

// A reach naming bytes past the record that holds it.  Nested in Wrap those
// bytes fall in the second eightbyte, which stays padding all the same.
// CIR-DAG: !rec_Nested = !cir.struct<"Nested" {bitfield !u32i, bitfield !cir.array<!cir.array<!u8i x 2> x 0>}>
// CIR-DAG: !rec_WideInner = !cir.struct<"WideInner" {bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 7> x 0>, pad !cir.array<!u8i x 7>}>

// A unit whose declarations stop at its own end carries no reach member.  An
// unnamed bit-field records one all the same, since the bytes its declared
// type covers hold user data even though it supplies no eightbyte class.
// CIR-DAG: !rec_UnnamedPastUnit = !cir.struct<"UnnamedPastUnit" {bitfield !u32i, bitfield !cir.array<!u8i x 0>, pad !cir.array<!u8i x 4>}>
// CIR-DAG: !rec_UnnamedWide = !cir.struct<"UnnamedWide" {bitfield !u8i, bitfield !cir.array<!cir.array<!u8i x 8> x 0>, pad !cir.array<!u8i x 7>}>
// CIR-DAG: !rec_NarrowUnit = !cir.struct<"NarrowUnit" {bitfield !u32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_SpillsPastUnit = !cir.struct<"SpillsPastUnit" {bitfield !u64i, bitfield !cir.array<!cir.array<!u8i x 3> x 0>, data !s32i}>
// CIR-DAG: !rec_UnnamedOnly = !cir.struct<"UnnamedOnly" {empty !u8i, bitfield !cir.array<!cir.array<!u8i x 3> x 0>}>
// CIR-DAG: !rec_UUnnamedOnly = !cir.union<"UUnnamedOnly" {empty !u8i, bitfield !cir.array<!cir.array<!u8i x 4> x 0>}>

// A union of plain bytes has no declaration to reach past its storage, so it
// keeps the narrow coercion.
// CIR-DAG: !rec_UBytes = !cir.union<"UBytes" {data !u8i, data !u8i}>

void take_bits(Bits v) {}

// CIR: cir.func {{.*}}@_Z9take_bits4Bits(%arg0: !u32i
// LLVM: define {{.*}}void @_Z9take_bits4Bits(i32 %{{.+}})

Bits return_bits(Bits v) { return v; }

// CIR: cir.func {{.*}}@_Z11return_bits4Bits(%arg0: !u32i{{.*}}) -> !u32i
// LLVM: define {{.*}}i32 @_Z11return_bits4Bits(i32 %{{.+}})

void take_outer(OuterNest v) {}

// CIR: cir.func {{.*}}@_Z10take_outer9OuterNest(%arg0: !u32i
// LLVM: define {{.*}}void @_Z10take_outer9OuterNest(i32 %{{.+}})

void take_holder(Holder v) {}

// CIR: cir.func {{.*}}@_Z11take_holder6Holder(%arg0: !s64i{{.*}}, %arg1: !u32i
// LLVM: define {{.*}}void @_Z11take_holder6Holder(i64 %{{.+}}, i32 %{{.+}})

// The reach also travels through an array element.  The element's unit is one
// byte, so without the reach the eightbyte would narrow to i8.
void take_arr_nest(ArrNest v) {}

// CIR: cir.func {{.*}}@_Z13take_arr_nest7ArrNest(%arg0: !u64i
// LLVM: define {{.*}}void @_Z13take_arr_nest7ArrNest(i64 %{{.+}})

// A reach may run past the end of the record that records it.  The bytes it
// names there belong to the enclosing record, and the classifier stops at the
// inner record's own size, so the eightbyte holding them stays padding.
void take_wrap(Wrap v) {}

// CIR: cir.func {{.*}}@_Z9take_wrap4Wrap(%arg0: !u64i
// LLVM: define {{.*}}void @_Z9take_wrap4Wrap(i64 %{{.+}})

// An unnamed bit-field supplies no eightbyte class, but the bytes its declared
// type covers are user data, so its reach counts.  Leaving it out would narrow
// these two to i32 and i8.
void take_unnamed_past_unit(UnnamedPastUnit v) {}
void take_unnamed_wide(UnnamedWide v) {}

// CIR: cir.func {{.*}}@_Z22take_unnamed_past_unit15UnnamedPastUnit(%arg0: !u64i
// LLVM: define {{.*}}void @_Z22take_unnamed_past_unit15UnnamedPastUnit(i64 %{{.+}})

// CIR: cir.func {{.*}}@_Z17take_unnamed_wide11UnnamedWide(%arg0: !u64i
// LLVM: define {{.*}}void @_Z17take_unnamed_wide11UnnamedWide(i64 %{{.+}})

// These two records differ only in the type their bit-field is declared with,
// so the reach is what widens the second coercion to the eightbyte it covers.
void take_narrow_unit(NarrowUnit v) {}
void take_wide_unit(WideUnit v) {}

// CIR: cir.func {{.*}}@_Z16take_narrow_unit10NarrowUnit(%arg0: !u32i
// LLVM: define {{.*}}void @_Z16take_narrow_unit10NarrowUnit(i32 %{{.+}})

// CIR: cir.func {{.*}}@_Z14take_wide_unit8WideUnit(%arg0: !u64i
// LLVM: define {{.*}}void @_Z14take_wide_unit8WideUnit(i64 %{{.+}})

// A reach and a zero-width bit-field in one record, which share a shape.
void take_named_zero_width(NamedZeroWidth v) {}

// CIR: cir.func {{.*}}@_Z21take_named_zero_width14NamedZeroWidth(%arg0: !u64i
// LLVM: define {{.*}}void @_Z21take_named_zero_width14NamedZeroWidth(i64 %{{.+}})

// The int here sits at an offset its own alignment does not divide, which
// sends the record to memory whatever the unit before it reaches.
void take_packed_bits(PackedBits v) {}

// CIR: cir.func {{.*}}@_Z16take_packed_bits10PackedBits(%arg0: !cir.ptr<!rec_PackedBits>
// LLVM-CIR: define {{.*}}void @_Z16take_packed_bits10PackedBits(ptr noalias noundef byval(%struct.PackedBits) align 8 %{{.+}})
// LLVM-OGCG: define {{.*}}void @_Z16take_packed_bits10PackedBits(ptr noundef byval(%struct.PackedBits) align 8 %{{.+}})

// The reach is measured from where the member sits, which is the unit's end.
// Measuring it from the unit's start instead would claim a further eightbyte
// of user data and widen the second half to i64.
void take_spills_past_unit(SpillsPastUnit v) {}

// CIR: cir.func {{.*}}@_Z21take_spills_past_unit14SpillsPastUnit(%arg0: !u64i{{.*}}, %arg1: !s32i
// LLVM: define {{.*}}void @_Z21take_spills_past_unit14SpillsPastUnit(i64 %{{.+}}, i32 %{{.+}})

// A unit of unnamed bit-fields records a reach without supplying data, which
// leaves the record empty for the ABI and drops the argument.
void take_unnamed_only(UnnamedOnly v) {}
void take_uunnamed_only(UUnnamedOnly v) {}

// CIR: cir.func {{.*}}@_Z17take_unnamed_only11UnnamedOnly()
// LLVM: define {{.*}}void @_Z17take_unnamed_only11UnnamedOnly()

// CIR: cir.func {{.*}}@_Z18take_uunnamed_only12UUnnamedOnly()
// LLVM: define {{.*}}void @_Z18take_uunnamed_only12UUnnamedOnly()

void take_ubits(UBits v) {}

// CIR: cir.func {{.*}}@_Z10take_ubits5UBits(%arg0: !u32i
// LLVM: define {{.*}}void @_Z10take_ubits5UBits(i32 %{{.+}})

void take_ubytes(UBytes v) {}

// CIR: cir.func {{.*}}@_Z11take_ubytes6UBytes(%arg0: !u8i
// LLVM: define {{.*}}void @_Z11take_ubytes6UBytes(i8 %{{.+}})

// A union holding both a data member and a bit-field whose declared type
// outruns it, so the reach rather than the widest member sizes the coercion.
void take_uwide_bits(UWideBits v) {}
UWideBits ret_uwide_bits(void) { UWideBits u; u.w = 1; return u; }

// CIR: cir.func {{.*}}@_Z15take_uwide_bits9UWideBits(%arg0: !u64i
// LLVM: define {{.*}}void @_Z15take_uwide_bits9UWideBits(i64 %{{.+}})

// CIR: cir.func {{.*}}@_Z14ret_uwide_bitsv() -> !u64i
// LLVM: define {{.*}}i64 @_Z14ret_uwide_bitsv()
