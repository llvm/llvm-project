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
// CIR-DAG:  #bfi_c = #cir.bitfield_info<name = "c", storageType = !u64i, size = 17, offset = 32, isSigned = true>
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
