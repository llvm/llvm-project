// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef union { int x : 3; } BitExtent;
typedef union { int x : 3; char c; } BitExtentPlusChar;
typedef union { long long x : 32; int y; } BitExtentWide;
typedef union { long long x : 32; float f; } BitFloatSibling;
typedef union { int x : 20; } BitArrayUnit;
typedef union { int x : 3; long long y : 40; } TwoBitUnits;
typedef union { int : 20; short s; } UnnamedBitExtent;
typedef union { int x : 3; int : 0; } ZeroWidthTail;
typedef struct { union { int x : 20; } u; int k; } WrapsUnion;
typedef union { long long x : 3; } __attribute__((packed)) BitExtentPacked;
typedef union { long long x : 3; } __attribute__((packed, aligned(2))) Overshoot;
typedef union { _BitInt(72) x : 40; } WideBitIntDecl;
typedef union { __int128 x : 40; } __attribute__((packed, aligned(8))) WideInt128Decl;
typedef union { int x : 20; } __attribute__((packed)) ArrayUnitPacked;
typedef union { int : 20; } UnnamedOnly;
typedef struct { BitExtent u; long long k; } Wrap16;
typedef struct { BitExtent u; char big[32]; } Wrap40;
typedef struct { BitExtent a[2]; } WrapArr;
typedef union { BitExtent inner; int y; } NestUnion;

// CIR-DAG: !rec_ZeroWidthTail = !cir.union<"ZeroWidthTail" {bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3>]>}, padding = {!cir.array<!u8i x 3>}>
// CIR-DAG: ![[PAIR_RET:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !s64i}>

// The access unit is one byte where the union is four, and the `int` the
// bit-field was declared with is what accounts for the rest.
void take_bit_extent(BitExtent u) {}
// CIR: cir.func{{.*}} @take_bit_extent(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_bit_extent(i32 %{{[^,)]+}})

void take_bit_extent_plus_char(BitExtentPlusChar u) {}
// CIR: cir.func{{.*}} @take_bit_extent_plus_char(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_bit_extent_plus_char(i32 %{{[^,)]+}})

// The declared `long long` reaches all eight bytes where the sibling `int` and
// the four-byte unit both stop at four.
void take_bit_extent_wide(BitExtentWide u) {}
// CIR: cir.func{{.*}} @take_bit_extent_wide(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_bit_extent_wide(i64 %{{[^,)]+}})

// The sibling classifies SSE and the unit INTEGER, and the merge takes
// INTEGER, so the declaration decides the size and not the class.
void take_bit_float_sibling(BitFloatSibling u) {}
// CIR: cir.func{{.*}} @take_bit_float_sibling(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_bit_float_sibling(i64 %{{[^,)]+}})

// A 20-bit field takes a three-byte unit, so the unit is an array rather than
// an integer.
void take_bit_array_unit(BitArrayUnit u) {}
// CIR: cir.func{{.*}} @take_bit_array_unit(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_bit_array_unit(i32 %{{[^,)]+}})

// Each variant of a union is its own access unit, so these two bit-fields do
// not share one and the widest declaration among them is what counts.
void take_two_bit_units(TwoBitUnits u) {}
// CIR: cir.func{{.*}} @take_two_bit_units(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_two_bit_units(i64 %{{[^,)]+}})

// An access unit of nothing but unnamed bit-fields still carries a declared
// type, and the union has no other member that reaches four bytes.
void take_unnamed_bit_extent(UnnamedBitExtent u) {}
// CIR: cir.func{{.*}} @take_unnamed_bit_extent(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_unnamed_bit_extent(i32 %{{[^,)]+}})

// A zero-width bit-field is no variant of the union, which is left with the
// named unit and its declaration.
void take_zero_width_tail(ZeroWidthTail u) {}
// CIR: cir.func{{.*}} @take_zero_width_tail(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_zero_width_tail(i32 %{{[^,)]+}})

// The union is reached as a struct member, so the struct's own eightbyte is
// what gets classified.
void take_wraps_union(WrapsUnion s) {}
// CIR: cir.func{{.*}} @take_wraps_union(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_wraps_union(i64 %{{[^,)]+}})

void take_wrap16(Wrap16 s) {}
// CIR: cir.func{{.*}} @take_wrap16(%arg0: !u64i loc{{.*}}, %arg1: !s64i loc{{.*}}) attributes
// LLVM: define{{.*}} void @take_wrap16(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

Wrap16 ret_wrap16(void) {
  Wrap16 s;
  s.u.x = 1;
  s.k = 2;
  return s;
}
// CIR: cir.func{{.*}} @ret_wrap16() -> ![[PAIR_RET]] attributes
// LLVM: define{{.*}} { i64, i64 } @ret_wrap16()

void take_wrap40(Wrap40 s) {}
// CIR: cir.func{{.*}} @take_wrap40(%arg0: !cir.ptr<!rec_Wrap40> {llvm.align = 8 : i64, llvm.byval = !rec_Wrap40, llvm.noundef} loc
// LLVM: define{{.*}} void @take_wrap40(ptr noundef byval(%struct.Wrap40) align 8 %{{[^,)]+}})

Wrap40 ret_wrap40(void) {
  Wrap40 s;
  s.u.x = 1;
  return s;
}
// CIR: cir.func{{.*}} @ret_wrap40(%arg0: !cir.ptr<!rec_Wrap40> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Wrap40, llvm.writable} loc
// LLVM: define{{.*}} void @ret_wrap40(ptr dead_on_unwind noalias writable sret(%struct.Wrap40) align 4 %{{[^,)]+}})

// The declared extent has to be found through the array.
void take_wraparr(WrapArr s) {}
// CIR: cir.func{{.*}} @take_wraparr(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_wraparr(i64 %{{[^,)]+}})

// And here through the outer union.
void take_nest_union(NestUnion u) {}
// CIR: cir.func{{.*}} @take_nest_union(%arg0: !u32i loc
// LLVM: define{{.*}} void @take_nest_union(i32 %{{[^,)]+}})

// Packed, so the one-byte unit covers this union without help from the
// declared `long long`.
void take_bit_extent_packed(BitExtentPacked u) {}
// CIR: cir.func{{.*}} @take_bit_extent_packed(%arg0: !u8i loc
// LLVM: define{{.*}} void @take_bit_extent_packed(i8 %{{[^,)]+}})

// Here neither the one-byte unit nor a sibling covers the two-byte union, so
// the declared `long long` is what covers it, overshooting by six bytes.
void take_overshoot(Overshoot u) {}
// CIR: cir.func{{.*}} @take_overshoot(%arg0: !u16i loc
// LLVM: define{{.*}} void @take_overshoot(i16 %{{[^,)]+}})

// A `_BitInt` declaration reaches as far as the width its alignment rounds it
// up to, which is eight bytes here rather than the seventy-two bits declared.
void take_wide_bitint_decl(WideBitIntDecl u) {}
// CIR: cir.func{{.*}} @take_wide_bitint_decl(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_wide_bitint_decl(i64 %{{[^,)]+}})

// The same 128-bit declaration the UBitWideDecl reject row carries, on a union
// small enough for the declared extent to be read at all.
void take_wide_int128_decl(WideInt128Decl u) {}
// CIR: cir.func{{.*}} @take_wide_int128_decl(%arg0: !u64i loc
// LLVM: define{{.*}} void @take_wide_int128_decl(i64 %{{[^,)]+}})

// A named unit covering its union on its own, so the second gate is satisfied
// without the declaration.
void take_array_unit_packed(ArrayUnitPacked u) {}
// CIR: cir.func{{.*}} @take_array_unit_packed(%arg0: !cir.int<u, 24> loc
// LLVM: define{{.*}} void @take_array_unit_packed(i24 %{{[^,)]+}})

// Here the unit covers the union on its own, without the declaration.
void take_unnamed_only(UnnamedOnly u) {}
// CIR: cir.func{{.*}} @take_unnamed_only(%arg0: !cir.int<u, 24> loc
// LLVM: define{{.*}} void @take_unnamed_only(i24 %{{[^,)]+}})

BitArrayUnit ret_bit_array_unit(void) {
  BitArrayUnit u;
  u.x = 1;
  return u;
}
// CIR: cir.func{{.*}} @ret_bit_array_unit() -> !u32i
// LLVM: define{{.*}} i32 @ret_bit_array_unit()

void call_bit_array_unit(void) { take_bit_array_unit(ret_bit_array_unit()); }
// CIR: cir.func{{.*}} @call_bit_array_unit()
// CIR:   cir.call @take_bit_array_unit(%{{.+}}) : (!u32i) -> ()
// LLVM: define{{.*}} void @call_bit_array_unit()
// LLVM:   call void @take_bit_array_unit(i32 %{{[^,)]+}})

void vsink(int n, ...);
void call_variadic(BitExtent u) { vsink(1, u); }
// CIR: cir.func{{.*}} @call_variadic(%arg0: !u32i loc
// CIR:   cir.call @vsink(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !u32i) -> ()
// LLVM: define{{.*}} void @call_variadic(i32 %{{[^,)]+}})
// LLVM:   call void (i32, ...) @vsink(i32 noundef 1, i32 %{{[^,)]+}})
