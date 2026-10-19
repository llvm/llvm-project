// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --implicit-check-not=i80 --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --implicit-check-not=i80 --input-file=%t.ll %s

typedef long double ld1 __attribute__((vector_size(16)));
typedef __int128 m1 __attribute__((vector_size(16)));
typedef long double ld2 __attribute__((vector_size(32)));
typedef __int128 m2 __attribute__((vector_size(32)));

// !, && and || on an x87 vector yield 128-bit integer elements.
void lnot(m1 *r, ld1 *v) { *r = !*v; }

// CIR-LABEL: cir.func{{.*}} @_Z4lnotPDv1_nPDv1_e(
// CIR: cir.vec.cmp(eq, %{{.+}}, %{{.+}}) : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.vector<1 x !s128i>

// LLVM-LABEL: define dso_local void @_Z4lnotPDv1_nPDv1_e(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM: %[[CMP:.+]] = fcmp oeq <1 x x86_fp80> %{{.+}}, zeroinitializer
// LLVM: %[[EXT:.+]] = sext <1 x i1> %[[CMP]] to <1 x i128>
// LLVM: store <1 x i128> %[[EXT]], ptr %{{.+}}, align 16

void lnot2(m2 *r, ld2 *v) { *r = !*v; }

// CIR-LABEL: cir.func{{.*}} @_Z5lnot2PDv2_nPDv2_e(
// CIR: cir.vec.cmp(eq, %{{.+}}, %{{.+}}) : !cir.vector<2 x !cir.long_double<!cir.f80>>, !cir.vector<2 x !s128i>

// LLVM-LABEL: define dso_local void @_Z5lnot2PDv2_nPDv2_e(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM: %[[CMP:.+]] = fcmp oeq <2 x x86_fp80> %{{.+}}, zeroinitializer
// LLVM: %[[EXT:.+]] = sext <2 x i1> %[[CMP]] to <2 x i128>
// LLVM: store <2 x i128> %[[EXT]], ptr %{{.+}}, align 32

void land(m1 *r, ld1 *a, ld1 *b) { *r = *a && *b; }

// CIR-LABEL: cir.func{{.*}} @_Z4landPDv1_nPDv1_eS2_(
// CIR: %[[A:.+]] = cir.vec.cmp(ne, %{{.+}}, %{{.+}}) : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.vector<1 x !s128i>
// CIR: %[[B:.+]] = cir.vec.cmp(ne, %{{.+}}, %{{.+}}) : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.vector<1 x !s128i>
// CIR: cir.and %[[A]], %[[B]] : !cir.vector<1 x !s128i>

// LLVM-LABEL: define dso_local void @_Z4landPDv1_nPDv1_eS2_(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM: %[[A:.+]] = fcmp une <1 x x86_fp80> %{{.+}}, zeroinitializer
// LLVMCIR: %[[A_EXT:.+]] = sext <1 x i1> %[[A]] to <1 x i128>
// LLVM: %[[B:.+]] = fcmp une <1 x x86_fp80> %{{.+}}, zeroinitializer
// LLVMCIR: %[[B_EXT:.+]] = sext <1 x i1> %[[B]] to <1 x i128>
// LLVMCIR: %[[RES:.+]] = and <1 x i128> %[[A_EXT]], %[[B_EXT]]
// OGCG: %[[AND:.+]] = and <1 x i1> %[[A]], %[[B]]
// OGCG: %[[RES:.+]] = sext <1 x i1> %[[AND]] to <1 x i128>
// LLVM: store <1 x i128> %[[RES]], ptr %{{.+}}, align 16

void lor(m1 *r, ld1 *a, ld1 *b) { *r = *a || *b; }

// CIR-LABEL: cir.func{{.*}} @_Z3lorPDv1_nPDv1_eS2_(
// CIR: %[[A:.+]] = cir.vec.cmp(ne, %{{.+}}, %{{.+}}) : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.vector<1 x !s128i>
// CIR: %[[B:.+]] = cir.vec.cmp(ne, %{{.+}}, %{{.+}}) : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.vector<1 x !s128i>
// CIR: cir.or %[[A]], %[[B]] : !cir.vector<1 x !s128i>

// LLVM-LABEL: define dso_local void @_Z3lorPDv1_nPDv1_eS2_(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM: %[[A:.+]] = fcmp une <1 x x86_fp80> %{{.+}}, zeroinitializer
// LLVMCIR: %[[A_EXT:.+]] = sext <1 x i1> %[[A]] to <1 x i128>
// LLVM: %[[B:.+]] = fcmp une <1 x x86_fp80> %{{.+}}, zeroinitializer
// LLVMCIR: %[[B_EXT:.+]] = sext <1 x i1> %[[B]] to <1 x i128>
// LLVMCIR: %[[RES:.+]] = or <1 x i128> %[[A_EXT]], %[[B_EXT]]
// OGCG: %[[OR:.+]] = or <1 x i1> %[[A]], %[[B]]
// OGCG: %[[RES:.+]] = sext <1 x i1> %[[OR]] to <1 x i128>
// LLVM: store <1 x i128> %[[RES]], ptr %{{.+}}, align 16
