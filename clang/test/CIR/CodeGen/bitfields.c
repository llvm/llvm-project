// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef struct {
  char a, b, c;
  unsigned bits : 3;
  unsigned more_bits : 4;
  unsigned still_more_bits : 7;
} A;

// CIR-DAG:  !rec_A = !cir.record<struct "A" packed padded {!s8i, !s8i, !s8i, !u16i, !cir.array<!u8i x 3>}>
// CIR-DAG:  #bfi_more_bits = #cir.bitfield_info<name = "more_bits", storage_type = !u16i, size = 4, offset = 3, is_signed = false>
// LLVM-DAG: %struct.A = type <{ i8, i8, i8, i16, [3 x i8] }>
// OGCG-DAG: %struct.A = type <{ i8, i8, i8, i16, [3 x i8] }>

typedef struct {
  int a : 4;
  int b : 5;
  int c;
} D;

// CIR-DAG:  !rec_D = !cir.record<struct "D" {!u16i, !s32i}>
// LLVM-DAG: %struct.D = type { i16, i32 }
// OGCG-DAG: %struct.D = type { i16, i32 }

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S;
// CIR-DAG:  #bfi_c = #cir.bitfield_info<name = "c", storage_type = !u64i, size = 17, offset = 32, is_signed = true>
// CIR-DAG:  !rec_S = !cir.record<struct "S" {!u64i, !u16i, !u32i}>
// LLVM-DAG: %struct.S = type { i64, i16, i32 }
// OGCG-DAG: %struct.S = type { i64, i16, i32 }

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T;

// CIR-DAG:  !rec_T = !cir.record<struct "T" {!u8i, !u32i}>
// LLVM-DAG: %struct.T = type { i8, i32 }
// OGCG-DAG: %struct.T = type { i8, i32 }

typedef struct {
    char a;
    char b;
    char c;

    // startOffset 24 bits, new storage from here
    int d: 2;
    int e: 2;
    int f: 4;
    int g: 25;
    int h: 3;
    int i: 4;
    int j: 3;
    int k: 8;

    int l: 14;
} U;

// CIR-DAG:  !rec_U = !cir.record<struct "U" packed {!s8i, !s8i, !s8i, !u8i, !u64i}>
// LLVM-DAG: %struct.U = type <{ i8, i8, i8, i8, i64 }>
// OGCG-DAG: %struct.U = type <{ i8, i8, i8, i8, i64 }>

void def() {
  A a;
  D d;
  S s;
  T t;
  U u;
}

int load_field(S* s) {
  return s->c;
}

// CIR: cir.func {{.*@load_field}}
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init]
// CIR:   [[TMP1:%.*]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   [[TMP2:%.*]] = cir.get_member [[TMP1]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_c, [[TMP2]] : !cir.ptr<!u64i>) -> !s32i

// LLVM: define dso_local i32 @load_field
// LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   [[TMP1:%.*]] = alloca i32, i64 1, align 4
// LLVM:   [[TMP2:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP3:%.*]] = getelementptr %struct.S, ptr [[TMP2]], i32 0, i32 0
// LLVM:   [[TMP4:%.*]] = load i64, ptr [[TMP3]], align 8
// LLVM:   [[TMP5:%.*]] = shl i64 [[TMP4]], 15
// LLVM:   [[TMP6:%.*]] = ashr i64 [[TMP5]], 47
// LLVM:   [[TMP7:%.*]] = trunc i64 [[TMP6]] to i32

// OGCG: define dso_local i32 @load_field
// OGCG:  [[TMP0:%.*]] = alloca ptr, align 8
// OGCG:  [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG:  [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 4
// OGCG:  [[TMP3:%.*]] = shl i64 [[TMP2]], 15
// OGCG:  [[TMP4:%.*]] = ashr i64 [[TMP3]], 47
// OGCG:  [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32

unsigned int load_field_unsigned(A* s) {
  return s->more_bits;
}

//CIR: cir.func dso_local @load_field_unsigned
//CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["s", init] {alignment = 8 : i64}
//CIR:   [[TMP1:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
//CIR:   [[TMP2:%.*]] = cir.get_member [[TMP1]][3] {name = "more_bits"} : !cir.ptr<!rec_A> -> !cir.ptr<!u16i>
//CIR:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_more_bits, [[TMP2]] : !cir.ptr<!u16i>) -> !u32i

//LLVM: define dso_local i32 @load_field_unsigned
//LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
//LLVM:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
//LLVM:   [[TMP2:%.*]] = getelementptr %struct.A, ptr [[TMP1]], i32 0, i32 3
//LLVM:   [[TMP3:%.*]] = load i16, ptr [[TMP2]], align 2
//LLVM:   [[TMP4:%.*]] = lshr i16 [[TMP3]], 3
//LLVM:   [[TMP5:%.*]] = and i16 [[TMP4]], 15
//LLVM:   [[TMP6:%.*]] = zext i16 [[TMP5]] to i32

//OGCG: define dso_local i32 @load_field_unsigned
//OGCG:   [[TMP0:%.*]] = alloca ptr, align 8
//OGCG:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
//OGCG:   [[TMP2:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[TMP1]], i32 0, i32 3
//OGCG:   [[TMP3:%.*]] = load i16, ptr [[TMP2]], align 1
//OGCG:   [[TMP4:%.*]] = lshr i16 [[TMP3]], 3
//OGCG:   [[TMP5:%.*]] = and i16 [[TMP4]], 15
//OGCG:   [[TMP6:%.*]] = zext i16 [[TMP5]] to i32

void store_field() {
  S s;
  s.e = 3;
}
// CIR: cir.func {{.*@store_field}}
// CIR:   [[TMP0:%.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>
// CIR:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:   [[TMP2:%.*]] = cir.get_member [[TMP0]][1] {name = "e"} : !cir.ptr<!rec_S> -> !cir.ptr<!u16i>
// CIR:   cir.set_bitfield(#bfi_e, [[TMP2]] : !cir.ptr<!u16i>, [[TMP1]] : !s32i)

// LLVM: define dso_local void @store_field()
// LLVM:   [[TMP0:%.*]] = alloca %struct.S, i64 1, align 4
// LLVM:   [[TMP1:%.*]] = getelementptr %struct.S, ptr [[TMP0]], i32 0, i32 1
// LLVM:   [[TMP2:%.*]] = load i16, ptr [[TMP1]], align 2
// LLVM:   [[TMP3:%.*]] = and i16 [[TMP2]], -32768
// LLVM:   [[TMP4:%.*]] = or i16 [[TMP3]], 3
// LLVM:   store i16 [[TMP4]], ptr [[TMP1]], align 2

// OGCG: define dso_local void @store_field()
// OGCG:   [[TMP0:%.*]] = alloca %struct.S, align 4
// OGCG:   [[TMP1:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[TMP0]], i32 0, i32 1
// OGCG:   [[TMP2:%.*]] = load i16, ptr [[TMP1]], align 4
// OGCG:   [[TMP3:%.*]] = and i16 [[TMP2]], -32768
// OGCG:   [[TMP4:%.*]] = or i16 [[TMP3]], 3
// OGCG:   store i16 [[TMP4]], ptr [[TMP1]], align 4

void store_bitfield_to_bitfield() {
  S s;
  s.a = s.c;
}

// CIR: cir.func {{.*@store_bitfield_to_bitfield}}
// CIR:   [[TMP0:%.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s"] {alignment = 4 : i64}
// CIR:   [[TMP1:%.*]] = cir.get_member [[TMP0]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP2:%.*]] = cir.get_bitfield(#bfi_c, [[TMP1]] : !cir.ptr<!u64i>) -> !s32i
// CIR:   [[TMP3:%.*]] = cir.get_member [[TMP0]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP4:%.*]] = cir.set_bitfield(#bfi_a, [[TMP3]] : !cir.ptr<!u64i>, [[TMP2]] : !s32i) -> !s32i

// LLVM: define dso_local void @store_bitfield_to_bitfield()
// LLVM:  [[TMP0:%.*]] = alloca %struct.S, i64 1, align 4
// LLVM:  [[TMP1:%.*]] = getelementptr %struct.S, ptr [[TMP0]], i32 0, i32 0
// LLVM:  [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 8
// LLVM:  [[TMP3:%.*]] = shl i64 [[TMP2]], 15
// LLVM:  [[TMP4:%.*]] = ashr i64 [[TMP3]], 47
// LLVM:  [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
// LLVM:  [[TMP6:%.*]] = getelementptr %struct.S, ptr [[TMP0]], i32 0, i32 0
// LLVM:  [[TMP7:%.*]] = zext i32 [[TMP5]] to i64
// LLVM:  [[TMP8:%.*]] = load i64, ptr [[TMP6]], align 8
// LLVM:  [[TMP9:%.*]] = and i64 [[TMP7]], 15
// LLVM:  [[TMP10:%.*]] = and i64 [[TMP8]], -16
// LLVM:  [[TMP11:%.*]] = or i64 [[TMP10]], [[TMP9]]
// LLVM:  store i64 [[TMP11]], ptr [[TMP6]], align 8
// LLVM:  [[TMP12:%.*]] = shl i64 [[TMP9]], 60
// LLVM:  [[TMP13:%.*]] = ashr i64 [[TMP12]], 60
// LLVM:  [[TMP15:%.*]] = trunc i64 [[TMP13]] to i32

// OGCG: define dso_local void @store_bitfield_to_bitfield()
// OGCG:  [[TMP0:%.*]] = alloca %struct.S, align 4
// OGCG:  [[TMP1:%.*]] = load i64, ptr [[TMP0]], align 4
// OGCG:  [[TMP2:%.*]] = shl i64 [[TMP1]], 15
// OGCG:  [[TMP3:%.*]] = ashr i64 [[TMP2]], 47
// OGCG:  [[TMP4:%.*]] = trunc i64 [[TMP3]] to i32
// OGCG:  [[TMP5:%.*]] = zext i32 [[TMP4]] to i64
// OGCG:  [[TMP6:%.*]] = load i64, ptr [[TMP0]], align 4
// OGCG:  [[TMP7:%.*]] = and i64 [[TMP5]], 15
// OGCG:  [[TMP8:%.*]] = and i64 [[TMP6]], -16
// OGCG:  [[TMP9:%.*]] = or i64 [[TMP8]], [[TMP7]]
// OGCG:  store i64 [[TMP9]], ptr [[TMP0]], align 4
// OGCG:  [[TMP10:%.*]] = shl i64 %bf.value, 60
// OGCG:  [[TMP11:%.*]] = ashr i64 [[TMP10]], 60
// OGCG:  [[TMP12:%.*]] = trunc i64 [[TMP11]] to i32

typedef struct {
  int a : 30;
  int volatile b : 8;
  int c;
} V;

void get_volatile(V* v) {
  v->b = 3;
}

// CIR: cir.func dso_local @get_volatile
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_V>, !cir.ptr<!cir.ptr<!rec_V>>, ["v", init] {alignment = 8 : i64}
// CIR:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:   [[TMP2:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_V>>, !cir.ptr<!rec_V>
// CIR:   [[TMP3:%.*]] = cir.get_member [[TMP2]][0] {name = "b"} : !cir.ptr<!rec_V> -> !cir.ptr<!u64i>
// CIR:   [[TMP4:%.*]] = cir.set_bitfield(#bfi_b, [[TMP3]] : !cir.ptr<!u64i>, [[TMP1]] : !s32i) {is_volatile} -> !s32i

// LLVM: define dso_local void @get_volatile
// LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP2:%.*]] = getelementptr %struct.V, ptr [[TMP1]], i32 0, i32 0
// LLVM:   [[TMP3:%.*]] = load volatile i64, ptr [[TMP2]], align 8
// LLVM:   [[TMP4:%.*]] = and i64 [[TMP3]], -1095216660481
// LLVM:   [[TMP5:%.*]] = or i64 [[TMP4]], 12884901888
// LLVM:   store volatile i64 [[TMP5]], ptr [[TMP2]], align 8

// OCGC: define dso_local void @get_volatile
// OCGC:   [[TMP0:%.*]] = alloca ptr, align 8
// OCGC:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// OCGC:   [[TMP2:%.*]] = load volatile i64, ptr [[TMP1]], align 4
// OCGC:   [[TMP3:%.*]] = and i64 [[TMP2]], -1095216660481
// OCGC:   [[TMP4:%.*]] = or i64 [[TMP3]], 12884901888
// OCGC:   store volatile i64 [[TMP4]], ptr [[TMP1]], align 4

void set_volatile(V* v) {
  v->b = 3;
}
//CIR: cir.func dso_local @set_volatile
//CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_V>, !cir.ptr<!cir.ptr<!rec_V>>, ["v", init] {alignment = 8 : i64}
//CIR:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
//CIR:   [[TMP2:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_V>>, !cir.ptr<!rec_V>
//CIR:   [[TMP3:%.*]] = cir.get_member [[TMP2]][0] {name = "b"} : !cir.ptr<!rec_V> -> !cir.ptr<!u64i>
//CIR:   [[TMP4:%.*]] = cir.set_bitfield(#bfi_b, [[TMP3]] : !cir.ptr<!u64i>, [[TMP1]] : !s32i) {is_volatile} -> !s32i

// LLVM: define dso_local void @set_volatile
// LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP2:%.*]] = getelementptr %struct.V, ptr [[TMP1]], i32 0, i32 0
// LLVM:   [[TMP3:%.*]] = load volatile i64, ptr [[TMP2]], align 8
// LLVM:   [[TMP4:%.*]] = and i64 [[TMP3]], -1095216660481
// LLVM:   [[TMP5:%.*]] = or i64 [[TMP4]], 12884901888
// LLVM:   store volatile i64 [[TMP5]], ptr [[TMP2]], align 8

// OGCG: define dso_local void @set_volatile
// OGCG:   [[TMP0:%.*]] = alloca ptr, align 8
// OGCG:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG:   [[TMP2:%.*]] = load volatile i64, ptr [[TMP1]], align 4
// OGCG:   [[TMP3:%.*]] = and i64 [[TMP2]], -1095216660481
// OGCG:   [[TMP4:%.*]] = or i64 [[TMP3]], 12884901888
// OGCG:   store volatile i64 [[TMP4]], ptr [[TMP1]], align 4

void unOp(S* s) {
  s->d++;
}

// CIR: cir.func {{.*@unOp}}
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init] {alignment = 8 : i64}
// CIR:   [[TMP1:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   [[TMP2:%.*]] = cir.get_member [[TMP1]][0] {name = "d"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!u64i>) -> !s32i
// CIR:   [[TMP4:%.*]] = cir.unary(inc, [[TMP3]]) nsw : !s32i, !s32i
// CIR:   cir.set_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!u64i>, [[TMP4]] : !s32i)

// LLVM: define {{.*@unOp}}
// LLVM:   [[TMP0:%.*]] = getelementptr %struct.S, ptr [[LOAD0:%.*]], i32 0, i32 0
// LLVM:   [[TMP1:%.*]] = load i64, ptr [[TMP0]], align 8
// LLVM:   [[TMP2:%.*]] = shl i64 [[TMP1]], 13
// LLVM:   [[TMP3:%.*]] = ashr i64 [[TMP2]], 62
// LLVM:   [[TMP4:%.*]] = trunc i64 [[TMP3]] to i32
// LLVM:   [[TMP5:%.*]] = add nsw i32 [[TMP4]], 1
// LLVM:   [[TMP6:%.*]] = zext i32 [[TMP5]] to i64
// LLVM:   [[TMP7:%.*]] = load i64, ptr [[TMP0]], align 8
// LLVM:   [[TMP8:%.*]] = and i64 [[TMP6]], 3
// LLVM:   [[TMP9:%.*]] = shl i64 [[TMP8]], 49
// LLVM:   [[TMP10:%.*]] = and i64 [[TMP7]], -1688849860263937
// LLVM:   [[TMP11:%.*]] = or i64 [[TMP10]], [[TMP9]]
// LLVM:   store i64 [[TMP11]], ptr [[TMP0]], align 8
// LLVM:   [[TMP12:%.*]] = shl i64 [[TMP8]], 62
// LLVM:   [[TMP13:%.*]] = ashr i64 [[TMP12]], 62
// LLVM:   [[TMP14:%.*]] = trunc i64 [[TMP13]] to i32

// OGCG: define {{.*@unOp}}
// OGCG:   [[TMP0:%.*]] = load ptr, ptr %s.addr, align 8
// OGCG:   [[TMP1:%.*]] = load i64, ptr [[TMP0]], align 4
// OGCG:   [[TMP2:%.*]] = shl i64 [[TMP1]], 13
// OGCG:   [[TMP3:%.*]] = ashr i64 [[TMP2]], 62
// OGCG:   [[TMP4:%.*]] = trunc i64 [[TMP3]] to i32
// OGCG:   [[TMP5:%.*]] = add nsw i32 [[TMP4]], 1
// OGCG:   [[TMP6:%.*]] = zext i32 [[TMP5]] to i64
// OGCG:   [[TMP7:%.*]] = load i64, ptr [[TMP0]], align 4
// OGCG:   [[TMP8:%.*]] = and i64 [[TMP6]], 3
// OGCG:   [[TMP9:%.*]] = shl i64 [[TMP8]], 49
// OGCG:   [[TMP10:%.*]] = and i64 [[TMP7]], -1688849860263937
// OGCG:   [[TMP11:%.*]] = or i64 [[TMP10]], [[TMP9]]
// OGCG:   store i64 [[TMP11]], ptr [[TMP0]], align 4
// OGCG:   [[TMP12:%.*]] = shl i64 [[TMP8]], 62
// OGCG:   [[TMP13:%.*]] = ashr i64 [[TMP12]], 62
// OGCG:   [[TMP14:%.*]] = trunc i64 [[TMP13]] to i32
