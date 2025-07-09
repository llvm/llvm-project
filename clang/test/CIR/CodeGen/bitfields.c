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
// CIR-DAG:  #bfi_more_bits = #cir.bitfield_info<name = "more_bits", storageType = !u16i, size = 4, offset = 3, isSigned = false>
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
// CIR-DAG:  #bfi_c = #cir.bitfield_info<name = "c", storageType = !u64i, size = 17, offset = 32, isSigned = true>
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
