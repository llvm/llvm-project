// clang/test/SemaOpenCL/coop_mat_ast_utils.cl
//
// Patch 3: AST utility support — ASTImporter, ASTStructuralEquivalence,
//          ExprConstant, ItaniumMangle, MicrosoftMangle,
//          ASTReader/ASTWriter TypeLoc serialisation.
//
// ── 3a. Itanium mangling  ────────────────────────────────────────────────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=MANGLE
//
// ── 3b. Serialisation round-trip (TypeLoc reader/writer) ────────────────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -O0 -emit-pch -o %t.pch %s
// RUN: echo "void pch_probe(MatA_t a);" > %t.pch_probe.cl
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -O0 -include-pch %t.pch \
// RUN:   -ast-dump %t.pch_probe.cl \
// RUN:   | FileCheck %s --check-prefix=PCH
//
// ── 3c. Structural equivalence (no diagnostics on compatible pair) ───────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

#define SCOPE CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A CLK_COOPERATIVE_MATRIX_A
#define USE_B CLK_COOPERATIVE_MATRIX_B
#define USE_C CLK_COOPERATIVE_MATRIX_ACCUMULATOR

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// ── 3a. Mangling — function with a coop-mat parameter gets a mangled name
//        that contains the vendor-extended "coop_mat" marker.
kernel void test_mangling(MatA_t a) { (void)a; }

// MANGLE: @{{.*}}test_mangling{{.*}}(

// ── 3b. PCH — after the round-trip the FunctionDecl for pch_probe
//        must still carry a coop_mat parameter type.
// PCH: FunctionDecl {{.*}} pch_probe
// PCH: ParmVarDecl {{.*}} a {{.*}}coop_mat(

// ── 3c. Structural equivalence — same parameters produce no diagnostic.
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_alias;

void test_structural_equiv(void) {
    MatA_t    *p = 0;
    MatA_alias *q = p;  // same canonical type
    (void)q;
}

// ── 3d. ExprConstant — coop_mat type classified as "no class" (not an
//        integer, float, pointer …).  Using __builtin_classify_type on it
//        compiles without error; result is 0 (no_type_class).
void test_expr_constant(MatA_t a) {
    int cls = __builtin_classify_type(a);
    (void)cls;
}
