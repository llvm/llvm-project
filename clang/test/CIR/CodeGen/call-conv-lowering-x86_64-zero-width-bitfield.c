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
typedef struct { int x; } __attribute__((aligned(16))) NoZeroWidth;
typedef struct { Tail t; } Nest;
struct { int x; unsigned : 0; } __attribute__((aligned(16))) gAnon;
typedef struct { long long a; int b; unsigned : 0; } Pair;
typedef struct { char c; int : 0; char d[24]; } TooBig;
typedef struct { char c; int : 0; long long y; } AfterZeroWidth;
typedef struct { char c; long long y; } NoZeroWidthPair;
typedef struct { double d; int : 32; } __attribute__((aligned(16))) UnnamedUnit;

// CIR-DAG: AfterZeroWidth = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>
// CIR-DAG: Mid = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>
// CIR-DAG: Pair = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [96], zero_width_bitfield_widths = [32]>
// CIR-DAG: Plain = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>
// CIR-DAG: Tail = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>
// CIR-DAG: TooBig = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>
// CIR-DAG: Twice = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32, 64], zero_width_bitfield_widths = [32, 32]>
// CIR-DAG: Wide = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [64], zero_width_bitfield_widths = [64]>
// CIR-DAG: anon.0 = #cir.record_layout<{{[^>]*}}zero_width_bitfield_offsets = [32], zero_width_bitfield_widths = [32]>

// A record that declares none carries no list, so record_align closes the
// attribute.  `Nest` declares none itself, the bit-field being the member's.
// CIR-DAG: Nest = #cir.record_layout<{{[^>]*}}record_align = 16>
// CIR-DAG: NoZeroWidth = #cir.record_layout<{{[^>]*}}record_align = 16>

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

// `s` reaches bit 48 and the second bit-field starts at 64, so the pair of
// entries above is what this case asserts rather than the argument type.
void take_twice(Twice t) {}
// CIR: cir.func{{.*}} @take_twice(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_twice(i64 %{{.+}})

void take_no_zero_width(NoZeroWidth n) {}
// CIR: cir.func{{.*}} @take_no_zero_width(%arg0: !s32i loc
// LLVM: define{{.*}} void @take_no_zero_width(i32 %{{.+}})

void take_nest(Nest n) {}
// CIR: cir.func{{.*}} @take_nest(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_nest(i64 %{{.+}})

void take_anon(__typeof__(gAnon) a) {}
// CIR: cir.func{{.*}} @take_anon(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_anon(i64 %{{.+}})

// The bit-field lands in the second eightbyte, which is classified and coerced
// on its own.
void take_pair(Pair p) {}
// CIR: cir.func{{.*}} @take_pair(%arg0: !s64i loc{{.*}}, %arg1: !u64i loc
// LLVM: define{{.*}} void @take_pair(i64 %{{.+}}, i64 %{{.+}})

// A data member sits past the bit-field, so the field list only stays in offset
// order if the synthesized entry is placed rather than appended.
void take_after_zero_width(AfterZeroWidth a) {}
// CIR: cir.func{{.*}} @take_after_zero_width(%arg0: !u64i loc{{.*}}, %arg1: !s64i loc
// LLVM: define{{.*}} void @take_after_zero_width(i64 %{{.+}}, i64 %{{.+}})

void take_no_zw(NoZeroWidthPair n) {}
// CIR: cir.func{{.*}} @take_no_zw(%arg0: !s8i loc{{.*}}, %arg1: !s64i loc
// LLVM: define{{.*}} void @take_no_zw(i8 %{{.+}}, i64 %{{.+}})

// The unit occupies four bytes after `d`.  Counting it as data would stop this
// being a single SSE value and pass it as a double plus an i32.
void take_unnamed_unit(UnnamedUnit u) {}
// CIR: cir.func{{.*}} @take_unnamed_unit(%arg0: !cir.double loc
// LLVM: define{{.*}} void @take_unnamed_unit(double %{{.+}})

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
// CIR: cir.func{{.*}} @ret_pair() -> !rec_anon_struct
// LLVM: define{{.*}} { i64, i64 } @ret_pair()
