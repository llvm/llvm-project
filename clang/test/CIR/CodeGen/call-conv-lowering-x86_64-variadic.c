// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

typedef struct { int x; int y; } Pair2;
typedef struct { long a; long b; } Pair16;
typedef struct { long a, b, c, d; } Big;
typedef struct { __int128 w; } Wide;
typedef struct { __int128 w; char c; } WideChar;

int vf(Pair2 p, ...);

// CIR: cir.func private @vf(!u64i, ...) -> !s32i

int call_scalar(Pair2 p, int a, double d) { return vf(p, a, d); }

// CIR-LABEL: cir.func {{.*}}@call_scalar(%arg0: !u64i loc({{.+}}), %arg1: !s32i {llvm.noundef} loc({{.+}}), %arg2: !cir.double {llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}, %{{.+}}) : (!u64i, !s32i {llvm.noundef}, !cir.double {llvm.noundef}) -> !s32i

// LLVM-LABEL: i32 @call_scalar(i64 %{{.+}}, i32 noundef %{{.+}}, double noundef %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i32 noundef %{{.+}}, double noundef %{{.+}})

// A two-eightbyte record at the ellipsis is flattened into two INTEGER
// registers while registers remain.
int call_small(Pair2 p, Pair16 q) { return vf(p, q); }

// CIR-LABEL: cir.func {{.*}}@call_small(%arg0: !u64i loc({{.+}}), %arg1: !s64i loc({{.+}}), %arg2: !s64i loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}, %{{.+}}) : (!u64i, !s64i, !s64i) -> !s32i

// LLVM-LABEL: i32 @call_small(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})

// Larger than two eightbytes is MEMORY regardless of register availability.
int call_big(Pair2 p, Big b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@call_big(%arg0: !u64i loc({{.+}}), %arg1: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noalias, llvm.noundef}) -> !s32i

// LLVM-CIR-LABEL: i32 @call_big(i64 %{{.+}}, ptr noalias noundef byval(%struct.Big) align 8 %{{.+}})
// LLVM-CIR: call i32 (i64, ...) @vf(i64 %{{.+}}, ptr noalias noundef byval(%struct.Big) align 8 %{{.+}})
// LLVM-OGCG-LABEL: i32 @call_big(i64 %{{.+}}, ptr noundef byval(%struct.Big) align 8 %{{.+}})
// LLVM-OGCG: call i32 (i64, ...) @vf(i64 %{{.+}}, ptr noundef byval(%struct.Big) align 8 %{{.+}})

// The same Pair16 that went to registers in call_small goes to memory here:
// the named parameter and four longs leave only one INTEGER register, and a
// two-eightbyte record cannot be split across a register and the stack.
int call_exhausted(Pair2 p, long a, long b, long c, long d, Pair16 q) {
  return vf(p, a, b, c, d, q);
}

// CIR-LABEL: cir.func {{.*}}@call_exhausted(%arg0: !u64i loc({{.+}}), %arg1: !s64i {llvm.noundef} loc({{.+}}), %arg2: !s64i {llvm.noundef} loc({{.+}}), %arg3: !s64i {llvm.noundef} loc({{.+}}), %arg4: !s64i {llvm.noundef} loc({{.+}}), %arg5: !cir.ptr<!rec_Pair16> {llvm.align = 8 : i64, llvm.byval = !rec_Pair16, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!u64i, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !cir.ptr<!rec_Pair16> {llvm.align = 8 : i64, llvm.byval = !rec_Pair16, llvm.noalias, llvm.noundef}) -> !s32i

// LLVM-CIR-LABEL: i32 @call_exhausted(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, ptr noalias noundef byval(%struct.Pair16) align 8 %{{.+}})
// LLVM-CIR: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, ptr noalias noundef byval(%struct.Pair16) align 8 %{{.+}})
// LLVM-OGCG-LABEL: i32 @call_exhausted(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, ptr noundef byval(%struct.Pair16) align 8 %{{.+}})
// LLVM-OGCG: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, ptr noundef byval(%struct.Pair16) align 8 %{{.+}})

// A 128-bit integer spans two eightbytes but is still passed whole.
int call_int128(Pair2 p, __int128 w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_int128(%arg0: !u64i loc({{.+}}), %arg1: !s128i {llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !s128i {llvm.noundef}) -> !s32i

// LLVM-LABEL: i32 @call_int128(i64 %{{.+}}, i128 noundef %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i128 noundef %{{.+}})

// Wrapping it in a record does not change the class: both eightbytes are
// INTEGER, so the record is coerced back to a bare i128.
int call_wide(Pair2 p, Wide w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_wide(%arg0: !u64i loc({{.+}}), %arg1: !s128i loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !s128i) -> !s32i

// LLVM-LABEL: i32 @call_wide(i64 %{{.+}}, i128 %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i128 %{{.+}})

// One trailing byte pushes the record past two eightbytes, so it goes to
// memory, and the 128-bit member keeps the slot at 16-byte alignment.
int call_wide_char(Pair2 p, WideChar w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_wide_char(%arg0: !u64i loc({{.+}}), %arg1: !cir.ptr<!rec_WideChar> {llvm.align = 16 : i64, llvm.byval = !rec_WideChar, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !cir.ptr<!rec_WideChar> {llvm.align = 16 : i64, llvm.byval = !rec_WideChar, llvm.noalias, llvm.noundef}) -> !s32i

// LLVM-CIR-LABEL: i32 @call_wide_char(i64 %{{.+}}, ptr noalias noundef byval(%struct.WideChar) align 16 %{{.+}})
// LLVM-CIR: call i32 (i64, ...) @vf(i64 %{{.+}}, ptr noalias noundef byval(%struct.WideChar) align 16 %{{.+}})
// LLVM-OGCG-LABEL: i32 @call_wide_char(i64 %{{.+}}, ptr noundef byval(%struct.WideChar) align 16 %{{.+}})
// LLVM-OGCG: call i32 (i64, ...) @vf(i64 %{{.+}}, ptr noundef byval(%struct.WideChar) align 16 %{{.+}})

// A _BitInt narrower than a register is extended at the ellipsis, same as a
// declared parameter.
int ell_bitint17(Pair2 p, _BitInt(17) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint17(%arg0: !u64i loc({{.+}}), %arg1: !cir.int<s, 17, bitint> {llvm.signext} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !cir.int<s, 17, bitint> {llvm.signext}) -> !s32i

// LLVM-CIR-LABEL: i32 @ell_bitint17(i64 %{{.+}}, i17 signext %{{.+}})
// LLVM-CIR: call i32 (i64, ...) @vf(i64 %{{.+}}, i17 signext %{{.+}})
// LLVM-OGCG-LABEL: i32 @ell_bitint17(i64 %{{.+}}, i17 noundef signext %{{.+}})
// LLVM-OGCG: call i32 (i64, ...) @vf(i64 %{{.+}}, i17 noundef signext %{{.+}})

// A width between 33 and 63 widens to one register.
int ell_bitint48(Pair2 p, _BitInt(48) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint48(%arg0: !u64i loc({{.+}}), %arg1: !u64i {llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !u64i {llvm.noundef}) -> !s32i

// LLVM-LABEL: i32 @ell_bitint48(i64 %{{.+}}, i64 noundef %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 noundef %{{.+}})

// A width between 65 and 127 coerces to a register pair, and both halves are
// passed through the ellipsis.
int ell_bitint96(Pair2 p, _BitInt(96) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint96(%arg0: !u64i loc({{.+}}), %arg1: !u64i loc({{.+}}), %arg2: !u64i loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}, %{{.+}}) : (!u64i, !u64i, !u64i) -> !s32i

// LLVM-CIR-LABEL: i32 @ell_bitint96(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// LLVM-CIR: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// LLVM-OGCG-LABEL: i32 @ell_bitint96(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}})
// LLVM-OGCG: call i32 (i64, ...) @vf(i64 %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}})

// At exactly 128 bits it stays in its natural type.
int ell_bitint128(Pair2 p, _BitInt(128) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint128(%arg0: !u64i loc({{.+}}), %arg1: !s128i_bitint {llvm.noundef} loc({{.+}})) -> !s32i
// CIR: cir.call @vf(%{{.+}}, %{{.+}}) : (!u64i, !s128i_bitint {llvm.noundef}) -> !s32i

// LLVM-LABEL: i32 @ell_bitint128(i64 %{{.+}}, i128 noundef %{{.+}})
// LLVM: call i32 (i64, ...) @vf(i64 %{{.+}}, i128 noundef %{{.+}})
