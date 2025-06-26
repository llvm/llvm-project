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
