// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct SSO {
  char *p;
  unsigned long len;
  union {
    char local[16];
    unsigned long capacity;
  };
};

// Inner union's tail padding must not bleed into the outer record.
// CIR: !rec_anon{{.*}} = !cir.union<"anon{{.*}}" padded {!cir.array<!s8i x 16>, !u64i}, padding = {!cir.array<!u8i x 8>}>
// CIR: !rec_SSO = !cir.struct<"SSO" {!cir.ptr<!s8i>, !u64i, !rec_anon{{.*}}}>

// LLVM: %struct.SSO = type { ptr, i64, %union.anon{{.*}} }
// LLVM: %union.anon{{.*}} = type { i64, [8 x i8] }

extern "C" SSO *last_of_three() {
  SSO *p = new SSO[3];
  return &p[2];
}

// Allocation is 3*sizeof(SSO)=96; per-element stride comes from struct size.
// LLVM-LABEL: define {{.*}}@last_of_three
// LLVM: call {{.*}}@_Znam(i64 noundef 96)
// LLVM: getelementptr{{.*}}%struct.SSO, ptr %{{.+}}, i64 2
