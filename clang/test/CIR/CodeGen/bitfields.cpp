// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S;
// CIR-DAG:  !rec_S = !cir.record<struct "S" {!u64i, !u16i, !u32i}>
// CIR-DAG:  #bfi_c = #cir.bitfield_info<name = "c", storage_type = !u64i, size = 17, offset = 32, is_signed = true>
// LLVM-DAG: %struct.S = type { i64, i16, i32 }
// OGCG-DAG: %struct.S = type { i64, i16, i32 }

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T;

// CIR-DAG:  !rec_T = !cir.record<struct "T" {!u8i, !u32i}>
// LLVM-DAG: %struct.T = type { i8, i32 }
// OGCG-DAG: %struct.T = type { i8, i32 }

void def() {
  S s;
  T t;
}

int load_field(S* s) {
  return s->c;
}
// CIR: cir.func dso_local @_Z10load_field
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init]
// CIR:   [[TMP1:%.*]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   [[TMP2:%.*]] = cir.get_member [[TMP1]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_c, [[TMP2]] : !cir.ptr<!u64i>) -> !s32i

// LLVM: define dso_local i32 @_Z10load_fieldP1S
// LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   [[TMP1:%.*]] = alloca i32, i64 1, align 4
// LLVM:   [[TMP2:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP3:%.*]] = getelementptr %struct.S, ptr [[TMP2]], i32 0, i32 0
// LLVM:   [[TMP4:%.*]] = load i64, ptr [[TMP3]], align 8
// LLVM:   [[TMP5:%.*]] = shl i64 [[TMP4]], 15
// LLVM:   [[TMP6:%.*]] = ashr i64 [[TMP5]], 47
// LLVM:   [[TMP7:%.*]] = trunc i64 [[TMP6]] to i32

// OGCG: define dso_local noundef i32 @_Z10load_fieldP1S
// OGCG:  [[TMP0:%.*]] = alloca ptr, align 8
// OGCG:  [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG:  [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 4
// OGCG:  [[TMP3:%.*]] = shl i64 [[TMP2]], 15
// OGCG:  [[TMP4:%.*]] = ashr i64 [[TMP3]], 47
// OGCG:  [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32

void store_field() {
  S s;
  s.a = 3;
}
// CIR: cir.func dso_local @_Z11store_field
// CIR:   [[TMP0:%.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>
// CIR:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:   [[TMP2:%.*]] = cir.get_member [[TMP0]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   cir.set_bitfield(#bfi_a, [[TMP2]] : !cir.ptr<!u64i>, [[TMP1]] : !s32i)

// LLVM: define dso_local void @_Z11store_fieldv
// LLVM:   [[TMP0:%.*]] = alloca %struct.S, i64 1, align 4
// LLVM:   [[TMP1:%.*]] = getelementptr %struct.S, ptr [[TMP0]], i32 0, i32 0
// LLVM:   [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 8
// LLVM:   [[TMP3:%.*]] = and i64 [[TMP2]], -16
// LLVM:   [[TMP4:%.*]] = or i64 [[TMP3]], 3
// LLVM:   store i64 [[TMP4]], ptr [[TMP1]], align 8

// OGCG: define dso_local void @_Z11store_fieldv()
// OGCG:   [[TMP0:%.*]] = alloca %struct.S, align 4
// OGCG:   [[TMP1:%.*]] = load i64, ptr [[TMP0]], align 4
// OGCG:   [[TMP2:%.*]] = and i64 [[TMP1]], -16
// OGCG:   [[TMP3:%.*]] = or i64 [[TMP2]], 3
// OGCG:   store i64 [[TMP3]], ptr [[TMP0]], align 4

void store_bitfield_to_bitfield(S* s) {
  s->a = s->b = 3;
}

// CIR: cir.func dso_local @_Z26store_bitfield_to_bitfieldP1S
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init] {alignment = 8 : i64}
// CIR:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:   [[TMP2:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   [[TMP3:%.*]] = cir.get_member [[TMP2]][0] {name = "b"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP4:%.*]] = cir.set_bitfield(#bfi_b, [[TMP3]] : !cir.ptr<!u64i>, [[TMP1]] : !s32i) -> !s32i
// CIR:   [[TMP5:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   [[TMP6:%.*]] = cir.get_member [[TMP5]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR:   [[TMP7:%.*]] = cir.set_bitfield(#bfi_a, [[TMP6]] : !cir.ptr<!u64i>, [[TMP4]] : !s32i) -> !s32i

// LLVM: define dso_local void @_Z26store_bitfield_to_bitfieldP1S
// LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP2:%.*]] = getelementptr %struct.S, ptr [[TMP1]], i32 0, i32 0
// LLVM:   [[TMP3:%.*]] = load i64, ptr [[TMP2]], align 8
// LLVM:   [[TMP4:%.*]] = and i64 [[TMP3]], -2147483633
// LLVM:   [[TMP5:%.*]] = or i64 [[TMP4]], 48
// LLVM:   store i64 [[TMP5]], ptr [[TMP2]], align 8
// LLVM:   [[TMP6:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM:   [[TMP7:%.*]] = getelementptr %struct.S, ptr [[TMP6]], i32 0, i32 0
// LLVM:   [[TMP8:%.*]] = load i64, ptr [[TMP7]], align 8
// LLVM:   [[TMP9:%.*]] = and i64 [[TMP8]], -16
// LLVM:   [[TMP10:%.*]] = or i64 [[TMP9]], 3
// LLVM:   store i64 [[TMP10]], ptr [[TMP7]], align 8

// OGCG: define dso_local void @_Z26store_bitfield_to_bitfieldP1S
// OGCG:   [[TMP0:%.*]] = alloca ptr, align 8
// OGCG:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG:   [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 4
// OGCG:   [[TMP3:%.*]] = and i64 [[TMP2]], -2147483633
// OGCG:   [[TMP4:%.*]] = or i64 [[TMP3]], 48
// OGCG:   store i64 [[TMP4]], ptr [[TMP1]], align 4
// OGCG:   [[TMP5:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG:   [[TMP6:%.*]] = load i64, ptr [[TMP5]], align 4
// OGCG:   [[TMP7:%.*]] = and i64 [[TMP6]], -16
// OGCG:   [[TMP8:%.*]] = or i64 [[TMP7]], 3
// OGCG:   store i64 [[TMP8]], ptr [[TMP5]], align 4
