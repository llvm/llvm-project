// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-SSE --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR-SSE --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG-SSE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-cir %s -o %t-avx.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-AVX --input-file=%t-avx.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-llvm %s -o %t-avx-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX,LLVM-CIR-AVX --input-file=%t-avx-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o %t-avx.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX,LLVM-OGCG-AVX --input-file=%t-avx.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -fclangir -emit-llvm %s -o %t-avx512-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX512 --input-file=%t-avx512-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm %s -o %t-avx512.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX512 --input-file=%t-avx512.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclang-abi-compat=23 -fclangir -emit-llvm %s -o %t-compat23-cir.ll
// RUN: FileCheck --check-prefix=LLVM-CIR-PINNED --input-file=%t-compat23-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclang-abi-compat=23 -emit-llvm %s -o %t-compat23.ll
// RUN: FileCheck --check-prefix=LLVM-OGCG-PINNED --input-file=%t-compat23.ll %s

// RUN: %clang_cc1 -triple x86_64-scei-ps4 -fclangir -emit-llvm %s -o %t-ps4-cir.ll
// RUN: FileCheck --check-prefix=LLVM-CIR-PINNED --input-file=%t-ps4-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-scei-ps4 -emit-llvm %s -o %t-ps4.ll
// RUN: FileCheck --check-prefix=LLVM-OGCG-PINNED --input-file=%t-ps4.ll %s

typedef float v4f __attribute__((vector_size(16)));
typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

// A 128-bit vector is at or below the native vector size at every AVX level,
// so it always passes in a register.
void take_v128(v4f v) { (void)v; }

// CIR: cir.func {{.*}}@take_v128(%arg0: !cir.vector<4 x !cir.float>{{.*}})
// LLVM: define dso_local void @take_v128(<4 x float> noundef %{{[^,)]+}})

// A 256-bit vector only reaches a register once the ABI level is AVX.  Below
// that it is passed byval, aligned to its size.
void take_v256(v8f v) { (void)v; }

// CIR-SSE: cir.func {{.*}}@take_v256(%arg0: !cir.ptr<!cir.vector<8 x !cir.float>> {{.*}}llvm.align = 32 : i64{{.*}}llvm.byval = !cir.vector<8 x !cir.float>{{.*}})
// CIR-AVX: cir.func {{.*}}@take_v256(%arg0: !cir.vector<8 x !cir.float>{{.*}})
// LLVM-CIR-SSE: define dso_local void @take_v256(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @take_v256(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @take_v256(<8 x float> noundef %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_v256(<8 x float> noundef %{{[^,)]+}})

// The register-versus-memory split applies to arguments only.  A 256-bit
// vector comes back in registers at every level.
v8f ret_v256(void) { v8f z = {0}; return z; }

// LLVM: define dso_local <8 x float> @ret_v256()

// Below AVX the caller has to build the byval slot, which is the memory copy a
// vector argument needs on a non-variadic call.
void call_v256(v8f v) { take_v256(v); }

// LLVM-CIR-SSE: define dso_local void @call_v256(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-CIR-SSE: call void @take_v256(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @call_v256(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-SSE: call void @take_v256(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @call_v256(<8 x float> noundef %{{[^,)]+}})
// LLVM-AVX: call void @take_v256(<8 x float> noundef %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @call_v256(<8 x float> noundef %{{[^,)]+}})
// LLVM-AVX512: call void @take_v256(<8 x float> noundef %{{[^,)]+}})

// A 512-bit vector needs AVX512 for the same treatment.
void take_v512(v16f v) { (void)v; }

// CIR-SSE: cir.func {{.*}}@take_v512(%arg0: !cir.ptr<!cir.vector<16 x !cir.float>> {{.*}}llvm.align = 64 : i64{{.*}}llvm.byval = !cir.vector<16 x !cir.float>{{.*}})
// CIR-AVX: cir.func {{.*}}@take_v512(%arg0: !cir.ptr<!cir.vector<16 x !cir.float>> {{.*}}llvm.align = 64 : i64{{.*}}llvm.byval = !cir.vector<16 x !cir.float>{{.*}})
// LLVM-CIR-SSE: define dso_local void @take_v512(ptr noalias noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @take_v512(ptr noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-CIR-AVX: define dso_local void @take_v512(ptr noalias noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-OGCG-AVX: define dso_local void @take_v512(ptr noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_v512(<16 x float> noundef %{{[^,)]+}})

// Disabling a feature the command line enabled is the only way a function's
// feature list carries a '-' entry, and the level still cannot fall below the
// module's, so this is byval only where the module itself lacks AVX512.
__attribute__((target("no-avx512f"))) void take_v512_no_avx512(v16f v) { (void)v; }

// LLVM-CIR-SSE: define dso_local void @take_v512_no_avx512(ptr noalias noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @take_v512_no_avx512(ptr noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-CIR-AVX: define dso_local void @take_v512_no_avx512(ptr noalias noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-OGCG-AVX: define dso_local void @take_v512_no_avx512(ptr noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_v512_no_avx512(<16 x float> noundef %{{[^,)]+}})

// A target attribute raises the level for one function, so these two classify
// their vector in a register at every configuration above.  An ABI older than
// the rule pins them back to the module's level, as does a target that opts out
// of the per-function rule entirely.
__attribute__((target("avx"))) void take_v256_tgt(v8f v) { (void)v; }

// CIR: cir.func {{.*}}@take_v256_tgt(%arg0: !cir.vector<8 x !cir.float>{{.*}})
// LLVM: define dso_local void @take_v256_tgt(<8 x float> noundef %{{[^,)]+}})
// LLVM-CIR-PINNED: define dso_local void @take_v256_tgt(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-PINNED: define dso_local void @take_v256_tgt(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})

__attribute__((target("avx512f"))) void take_v512_tgt(v16f v) { (void)v; }

// CIR: cir.func {{.*}}@take_v512_tgt(%arg0: !cir.vector<16 x !cir.float>{{.*}})
// LLVM: define dso_local void @take_v512_tgt(<16 x float> noundef %{{[^,)]+}})
// LLVM-CIR-PINNED: define dso_local void @take_v512_tgt(ptr noalias noundef byval(<16 x float>) align 64 %{{[^,)]+}})
// LLVM-OGCG-PINNED: define dso_local void @take_v512_tgt(ptr noundef byval(<16 x float>) align 64 %{{[^,)]+}})

// A call site has to agree with the callee it resolves to.  Both sides carry
// the attribute here, which is the case classic accepts.  A caller whose level
// disagrees with its callee is diagnosed by checkFunctionCallABI in classic
// CodeGen, and nothing diagnoses it here yet.
__attribute__((target("avx"))) void call_v256_tgt(v8f v) { take_v256_tgt(v); }

// LLVM: define dso_local void @call_v256_tgt(<8 x float> noundef %{{[^,)]+}})
// LLVM: call void @take_v256_tgt(<8 x float> noundef %{{[^,)]+}})
// LLVM-CIR-PINNED: define dso_local void @call_v256_tgt(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-CIR-PINNED: call void @take_v256_tgt(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-PINNED: define dso_local void @call_v256_tgt(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-PINNED: call void @take_v256_tgt(ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})

// A callee resolved at run time has no features of its own, so the level comes
// from the function containing the call.  The pair differs only in the
// attribute, so it is the attribute that has to move the argument.
typedef void (*v8f_fn)(v8f);
__attribute__((target("avx"))) void call_indirect(v8f_fn p, v8f v) { p(v); }

// LLVM: define dso_local void @call_indirect(ptr noundef %{{[^,)]+}}, <8 x float> noundef %{{[^,)]+}})
// LLVM: call void %{{[0-9]+}}(<8 x float> noundef %{{[^,)]+}})
// LLVM-CIR-PINNED: define dso_local void @call_indirect(ptr noundef %{{[^,)]+}}, ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-PINNED: define dso_local void @call_indirect(ptr noundef %{{[^,)]+}}, ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})

void call_indirect_plain(v8f_fn p, v8f v) { p(v); }

// LLVM-CIR-SSE: define dso_local void @call_indirect_plain(ptr noundef %{{[^,)]+}}, ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-CIR-SSE: call void %{{[0-9]+}}(ptr noalias noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @call_indirect_plain(ptr noundef %{{[^,)]+}}, ptr noundef byval(<8 x float>) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @call_indirect_plain(ptr noundef %{{[^,)]+}}, <8 x float> noundef %{{[^,)]+}})
// LLVM-AVX: call void %{{[0-9]+}}(<8 x float> noundef %{{[^,)]+}})

// Only a named argument can reach a register, so the struct that passes
// directly as a declared parameter goes to memory at the ellipsis.  This is
// the rule that reads the declared-parameter boundary, and it is observable
// only where the level admits the vector.
typedef struct { v8f v; } StructWideVector;
int variadic(const char *f, ...);
void named_swv(StructWideVector s) { (void)s; }
void pass_swv(StructWideVector s) { variadic("x", s); }

// LLVM-AVX: define dso_local void @named_swv(<8 x float> %{{[^,)]+}})
// LLVM-AVX: define dso_local void @pass_swv(<8 x float> %{{[^,)]+}})
// LLVM-CIR-AVX: call i32 (ptr, ...) @variadic(ptr noundef @.str, ptr noalias noundef byval(%struct.StructWideVector) align 32 %{{[^,)]+}})
// LLVM-OGCG-AVX: call i32 (ptr, ...) @variadic(ptr noundef @.str, ptr noundef byval(%struct.StructWideVector) align 32 %{{[^,)]+}})

// A union whose widest member is a 256-bit vector is classified from that
// member, so it reaches registers once the level admits the vector.
union UnionWideVector { v8f v; float f; };
void take_union_wide_vector(union UnionWideVector u) { (void)u; }

// LLVM-CIR-SSE: define dso_local void @take_union_wide_vector(ptr noalias noundef byval(%union.UnionWideVector) align 32 %{{[^,)]+}})
// LLVM-OGCG-SSE: define dso_local void @take_union_wide_vector(ptr noundef byval(%union.UnionWideVector) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @take_union_wide_vector(<4 x double> %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_union_wide_vector(<4 x double> %{{[^,)]+}})
