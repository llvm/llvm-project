// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

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

struct SizeAlignmentMismatchUnion {
  union { char c; int i; };
};
// CIR-DAG: !rec_SizeAlignmentMismatchUnion = !cir.record<struct "SizeAlignmentMismatchUnion" {![[ANON_UNION2:.*]]}
// CIR-DAG: ![[ANON_UNION2]] = !cir.record<union {{.*}} {!s8i, !s32i}>
// LLVM-DAG: %struct.SizeAlignmentMismatchUnion = type { %[[ANON_UNION2:.*]] }
// LLVM-DAG: %[[ANON_UNION2]] = type { i32 } 
struct ContainsSizeAlignmentMismatchUnion {
  SizeAlignmentMismatchUnion u;
  int thing;
};
// CIR-DAG: !rec_ContainsSizeAlignmentMismatchUnion = !cir.record<struct "ContainsSizeAlignmentMismatchUnion" {!rec_SizeAlignmentMismatchUnion, !s32i}>
// LLVM-DAG: %struct.ContainsSizeAlignmentMismatchUnion =  type { %struct.SizeAlignmentMismatchUnion, i32 }

union alignas(32) AlignedUnion {
  int n[4];
  long long ll;
};
// CIR-DAG: !rec_AlignedUnion = !cir.record<union "AlignedUnion" padded {!cir.array<!s32i x 4>, !s64i, !cir.array<!u8i x 24>}>
// LLVM-DAG: %union.AlignedUnion = type { i64, [24 x i8] }

struct ContainsAlignedUnion {
  AlignedUnion au;
  int thing;
};
// FIXME: We currently don't consider 'alignas' in our MLIR size/alignment
// calculations, which we probably should. The result is that the padding is 
// incorrect here. The 16xi8 shouldn't be present in either the CIR or LLVMCIR
// lines.
// CIR-DAG: !rec_ContainsAlignedUnion = !cir.record<struct "ContainsAlignedUnion" padded {!rec_AlignedUnion, !cir.array<!u8i x 16>, !s32i, !cir.array<!u8i x 28>}>
// LLVMCIR-DAG: %struct.ContainsAlignedUnion = type { %union.AlignedUnion, [16 x i8], i32, [28 x i8] }
// OGCG-DAG: %struct.ContainsAlignedUnion = type { %union.AlignedUnion, i32, [28 x i8] }

void use() {
  ContainsHasAnonUnion u;
  ContainsHasEmptyUnion u2;
  ContainsSizeAlignmentMismatchUnion u3;
  ContainsAlignedUnion u4;
}
// CIR: cir.func {{.*}}@_Z3usev()
// CIR: cir.alloca !rec_ContainsHasAnonUnion, !cir.ptr<!rec_ContainsHasAnonUnion>, ["u"] {alignment = 8 : i64}
// CIR: cir.alloca !rec_ContainsHasEmptyUnion, !cir.ptr<!rec_ContainsHasEmptyUnion>, ["u2"] {alignment = 4 : i64}
// CIR: cir.alloca !rec_ContainsSizeAlignmentMismatchUnion, !cir.ptr<!rec_ContainsSizeAlignmentMismatchUnion>, ["u3"] {alignment = 4 : i64}
// CIR: cir.alloca !rec_ContainsAlignedUnion, !cir.ptr<!rec_ContainsAlignedUnion>, ["u4"] {alignment = 32 : i64}

// LLVM-LABEL: define {{.*}}@_Z3usev()
// LLVM:   alloca %struct.ContainsHasAnonUnion, {{.*}}align 8
// LLVM:   alloca %struct.ContainsHasEmptyUnion, {{.*}}align 4
// LLVM:   alloca %struct.ContainsSizeAlignmentMismatchUnion, {{.*}}align 4
// LLVM:   alloca %struct.ContainsAlignedUnion, {{.*}}align 32
