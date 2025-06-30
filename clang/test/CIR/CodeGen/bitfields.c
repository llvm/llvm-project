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
