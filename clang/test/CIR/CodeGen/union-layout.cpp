// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct HasAnonUnion {
  union {
    int buf[4];
    long long cap;
  };
};
// CIR-DAG: !rec_HasAnonUnion = !cir.record<struct "HasAnonUnion" {![[ANON_UNION:.*]]}>
// LLVM-DAG: %struct.HasAnonUnion = type { %[[ANON_UNION:.*]] }

// CIR-DAG: ![[ANON_UNION]] = !cir.record<union {{.*}} padded {!cir.array<!s32i x 4>, !s64i, !cir.array<!u8i x 8>}>
// LLVM-DAG: %[[ANON_UNION]] = type { i64, [8 x i8] }

struct ContainsHasAnonUnion {
  HasAnonUnion hau;
  int thing;
};
// CIR-DAG: !rec_ContainsHasAnonUnion = !cir.record<struct "ContainsHasAnonUnion" {!rec_HasAnonUnion, !s32i}>
// LLVM-DAG: %struct.ContainsHasAnonUnion = type { %struct.HasAnonUnion, i32 }

struct HasEmptyUnion {
  union {
  } u;
};

struct ContainsHasEmptyUnion {
  HasEmptyUnion heu;
  int thing;
};
// CIR-DAG: !rec_ContainsHasEmptyUnion = !cir.record<struct "ContainsHasEmptyUnion" padded {!cir.array<!u8i x 4>, !s32i}>
// LLVM-DAG: %struct.ContainsHasEmptyUnion = type { [4 x i8], i32 }


void use() {
  ContainsHasAnonUnion u;
  ContainsHasEmptyUnion u2;
}
// CIR: cir.func {{.*}}@_Z3usev()
// CIR: cir.alloca !rec_ContainsHasAnonUnion, !cir.ptr<!rec_ContainsHasAnonUnion>, ["u"] {alignment = 8 : i64}
// CIR: cir.alloca !rec_ContainsHasEmptyUnion, !cir.ptr<!rec_ContainsHasEmptyUnion>, ["u2"] {alignment = 4 : i64}

// LLVM-LABEL: define {{.*}}@_Z3usev()
// LLVM:   alloca %struct.ContainsHasAnonUnion, {{.*}}align 8
// LLVM:   alloca %struct.ContainsHasEmptyUnion, {{.*}}align 4
