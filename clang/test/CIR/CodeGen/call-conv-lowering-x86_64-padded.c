// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

typedef struct { int x; } __attribute__((aligned(8))) HalfPad;
typedef struct { int x; } __attribute__((aligned(16))) OvInt;
typedef struct { char c; int i __attribute__((aligned(8))); } MidPad;
typedef struct { float f; } __attribute__((aligned(16))) OvFloat;
typedef struct { float a; float b __attribute__((aligned(8))); } TwoFloat;
typedef struct { int a; int b __attribute__((aligned(16))); } BigPad;
typedef struct { float a; float b; } __attribute__((aligned(16))) TwoFSame;
typedef struct { _Complex float c; } __attribute__((aligned(16))) CplxPad;

// One eightbyte in each register class.
typedef struct { int a; float b __attribute__((aligned(8))); } MixPad;

typedef struct { OvInt o; } NestOv;
typedef struct { OvInt a[2]; } ArrOv;

typedef struct { OvInt a[1]; } ArrOne;

struct E {};
typedef struct { struct E e; int i; } __attribute__((aligned(16))) PadWithEmpty;

// CIR-DAG: !rec_HalfPad = !cir.struct<"HalfPad" {data !s32i, pad !cir.array<!u8i x 4>}>
// CIR-DAG: !rec_OvInt = !cir.struct<"OvInt" {data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_MidPad = !cir.struct<"MidPad" {data !s8i, pad !cir.array<!u8i x 7>, data !s32i, pad !cir.array<!u8i x 4>}>
// CIR-DAG: !rec_OvFloat = !cir.struct<"OvFloat" {data !cir.float, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_TwoFloat = !cir.struct<"TwoFloat" {data !cir.float, pad !cir.array<!u8i x 4>, data !cir.float, pad !cir.array<!u8i x 4>}>
// CIR-DAG: !rec_BigPad = !cir.struct<"BigPad" {data !s32i, pad !cir.array<!u8i x 12>, data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_NestOv = !cir.struct<"NestOv" {data !rec_OvInt}>
// CIR-DAG: !rec_ArrOv = !cir.struct<"ArrOv" {data !cir.array<!rec_OvInt x 2>}>
// CIR-DAG: !rec_PadWithEmpty = !cir.struct<"PadWithEmpty" {empty !rec_E, data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_TwoFSame = !cir.struct<"TwoFSame" {data !cir.float, data !cir.float, pad !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_CplxPad = !cir.struct<"CplxPad" {data !cir.complex<!cir.float>, pad !cir.array<!u8i x 8>}>
// CIR-DAG: !rec_ArrOne = !cir.struct<"ArrOne" {data !cir.array<!rec_OvInt x 1>}>
// CIR-DAG: !rec_MixPad = !cir.struct<"MixPad" {data !s32i, pad !cir.array<!u8i x 4>, data !cir.float, pad !cir.array<!u8i x 4>}>

// Anonymous coercion records are numbered in print order, so capture them.
// CIR-DAG: ![[F64F32:rec_anon_struct[0-9]*]] = !cir.struct<{data !cir.double, data !cir.float}>
// CIR-DAG: ![[I64I32:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !s32i}>
// CIR-DAG: ![[I64F32:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !cir.float}>

int take_half(HalfPad h) { return h.x; }
HalfPad ret_half(int v) { HalfPad h = {v}; return h; }

// CIR: cir.func{{.*}} @take_half(%arg0: !s32i loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_half(%arg0: !s32i {llvm.noundef} loc{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_half(i32 %{{.+}})
// LLVM: define dso_local i32 @ret_half(i32 noundef %{{.+}})

// The second eightbyte is entirely padding, so one register still carries it.
int take_ov(OvInt o) { return o.x; }
OvInt ret_ov(int v) { OvInt o = {v}; return o; }

// CIR: cir.func{{.*}} @take_ov(%arg0: !s32i loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_ov(%arg0: !s32i {llvm.noundef} loc{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_ov(i32 %{{.+}})
// LLVM: define dso_local i32 @ret_ov(i32 noundef %{{.+}})

// Each eightbyte carries one member, so the argument flattens into two.
int take_mid(MidPad m) { return m.i; }
MidPad ret_mid(int v) { MidPad m = {0, v}; return m; }

// CIR: cir.func{{.*}} @take_mid(%arg0: !u64i loc{{.*}}, %arg1: !s32i loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_mid(%arg0: !s32i {llvm.noundef} loc{{.*}}) -> ![[I64I32]]
// LLVM: define dso_local i32 @take_mid(i64 %{{.+}}, i32 %{{.+}})
// LLVM: define dso_local { i64, i32 } @ret_mid(i32 noundef %{{.+}})

float take_ovf(OvFloat o) { return o.f; }
OvFloat ret_ovf(float v) { OvFloat o = {v}; return o; }

// CIR: cir.func{{.*}} @take_ovf(%arg0: !cir.float loc{{.*}}) -> !cir.float
// CIR: cir.func{{.*}} @ret_ovf(%arg0: !cir.float {llvm.noundef} loc{{.*}}) -> !cir.float
// LLVM: define dso_local float @take_ovf(float %{{.+}})
// LLVM: define dso_local float @ret_ovf(float noundef %{{.+}})

// The low eightbyte holds one float and four padding bytes, which the SSE
// rules widen to a double rather than packing two floats.
float take_twof(TwoFloat t) { return t.b; }
TwoFloat ret_twof(float v) { TwoFloat t = {v, v}; return t; }

// CIR: cir.func{{.*}} @take_twof(%arg0: !cir.double loc{{.*}}, %arg1: !cir.float loc{{.*}}) -> !cir.float
// CIR: cir.func{{.*}} @ret_twof(%arg0: !cir.float {llvm.noundef} loc{{.*}}) -> ![[F64F32]]
// LLVM: define dso_local float @take_twof(double %{{.+}}, float %{{.+}})
// LLVM: define dso_local { double, float } @ret_twof(float noundef %{{.+}})

// Past two eightbytes, so MEMORY, and the record's declared alignment reaches
// the byval and sret attributes rather than the alignment of its members.
int take_big(BigPad b) { return b.b; }
BigPad ret_big(int v) { BigPad b = {0, v}; return b; }

// CIR: cir.func{{.*}} @take_big(%arg0: !cir.ptr<!rec_BigPad> {llvm.align = 16 : i64, llvm.byval = !rec_BigPad, llvm.noalias, llvm.noundef} loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_big(%arg0: !cir.ptr<!rec_BigPad> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_BigPad, llvm.writable} loc{{.*}}, %arg1: !s32i {llvm.noundef} loc{{.*}})
// CIR emits noalias on a byval argument where classic does not, here and
// wherever else this file splits a byval line by backend.
// LLVM-CIR: define dso_local i32 @take_big(ptr noalias noundef byval(%struct.BigPad) align 16 %{{.+}})
// LLVM-OGCG: define dso_local i32 @take_big(ptr noundef byval(%struct.BigPad) align 16 %{{.+}})
// LLVM: define dso_local void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.BigPad) align 16 %{{.+}}, i32 noundef %{{.+}})

// A padded member keeps its register coercion, while two of them outgrow two
// eightbytes and go to memory.
int take_nest(NestOv n) { return n.o.x; }
int take_arr(ArrOv a) { return a.a[1].x; }

// CIR: cir.func{{.*}} @take_nest(%arg0: !s32i loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @take_arr(%arg0: !cir.ptr<!rec_ArrOv> {llvm.align = 16 : i64, llvm.byval = !rec_ArrOv, llvm.noalias, llvm.noundef} loc{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_nest(i32 %{{.+}})
// LLVM-CIR: define dso_local i32 @take_arr(ptr noalias noundef byval(%struct.ArrOv) align 16 %{{.+}})
// LLVM-OGCG: define dso_local i32 @take_arr(ptr noundef byval(%struct.ArrOv) align 16 %{{.+}})

// The empty member contributes no field and does not shift the int.
int take_pwe(PadWithEmpty p) { return p.i; }
PadWithEmpty ret_pwe(int v) { PadWithEmpty p = {{}, v}; return p; }

// CIR: cir.func{{.*}} @take_pwe(%arg0: !s32i loc{{.*}}) -> !s32i
// CIR: cir.func{{.*}} @ret_pwe(%arg0: !s32i {llvm.noundef} loc{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_pwe(i32 %{{.+}})
// LLVM: define dso_local i32 @ret_pwe(i32 noundef %{{.+}})

// Both floats share the low eightbyte, so the SSE rules pack them into a vector
// rather than widening to a double as they do for TwoFloat above.
float take_2fsame(TwoFSame t) { return t.a; }
TwoFSame ret_2fsame(float v) { TwoFSame t = {v, v}; return t; }

// CIR: cir.func{{.*}} @take_2fsame(%arg0: !cir.vector<2 x !cir.float> loc{{.*}}) -> !cir.float
// CIR: cir.func{{.*}} @ret_2fsame(%arg0: !cir.float {llvm.noundef} loc{{.*}}) -> !cir.vector<2 x !cir.float>
// LLVM: define dso_local float @take_2fsame(<2 x float> %{{.+}})
// LLVM: define dso_local <2 x float> @ret_2fsame(float noundef %{{.+}})

// A _Complex member reaches the same vector coercion under padding.
_Complex float take_cplx(CplxPad c) { return c.c; }

// CIR: cir.func{{.*}} @take_cplx(%arg0: !cir.vector<2 x !cir.float> loc{{.*}}) -> !cir.vector<2 x !cir.float>
// LLVM: define dso_local <2 x float> @take_cplx(<2 x float> %{{.+}})

float take_mix(MixPad m) { return m.b; }
MixPad ret_mix(int v) { MixPad m = {v, 1.0f}; return m; }

// CIR: cir.func{{.*}} @take_mix(%arg0: !u64i loc{{.*}}, %arg1: !cir.float loc{{.*}}) -> !cir.float
// LLVM: define dso_local float @take_mix(i64 %{{.+}}, float %{{.+}})
// CIR: cir.func{{.*}} @ret_mix(%arg0: !s32i {llvm.noundef} loc{{.*}}) -> ![[I64F32]]
// LLVM: define dso_local { i64, float } @ret_mix(i32 noundef %{{.+}})

// One element, so the element's own padding is classified rather than the whole
// record going to memory.
int take_arrone(ArrOne a) { return a.a[0].x; }

// CIR: cir.func{{.*}} @take_arrone(%arg0: !s32i loc{{.*}}) -> !s32i
// LLVM: define dso_local i32 @take_arrone(i32 %{{.+}})
