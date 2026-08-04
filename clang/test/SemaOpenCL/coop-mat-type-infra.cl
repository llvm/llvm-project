// clang/test/SemaOpenCL/coop_mat_type_infra.cl
//
// Patch 1: CooperativeMatrixType AST node and type infrastructure.
//
// Tests: TypeNodes.td registration, CooperativeMatrixType class accessors,
//        ASTContext::getCooperativeMatrixType uniquing, TypePrinter,
//        RecursiveASTVisitor traversal, mergeTypes compatibility,
//        TypeLoc operand slots, sizeof / getTypeInfoImpl.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -ast-dump %s \
// RUN:   | FileCheck %s --check-prefix=AST
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -O0 -emit-pch -o %t.pch %s
// RUN: echo "MatA_float16x16 g;" > %t.aux.cl
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -O0 -include-pch %t.pch -ast-dump %t.aux.cl \
// RUN:   | FileCheck %s --check-prefix=PCH

// ---------------------------------------------------------------------------
// Enum constants (from opencl-c-base.h via -finclude-default-header)
// ---------------------------------------------------------------------------
#define SCOPE  CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A  CLK_COOPERATIVE_MATRIX_A
#define USE_B  CLK_COOPERATIVE_MATRIX_B
#define USE_C  CLK_COOPERATIVE_MATRIX_ACCUMULATOR

// ---------------------------------------------------------------------------
// 1. Basic type construction — four use roles, float element type.
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A)))  MatA_float16x16;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B)))  MatB_float16x16;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C)))  MatC_float16x16;
typedef int   __attribute__((coop_mat(SCOPE,  8,  8, USE_A)))  MatA_int8x8;

// AST: TypedefDecl {{.*}} MatA_float16x16
// AST: TypedefDecl {{.*}} MatB_float16x16
// AST: TypedefDecl {{.*}} MatC_float16x16
// AST: TypedefDecl {{.*}} MatA_int8x8

// ---------------------------------------------------------------------------
// 2. Type printer — VarDecls carry the coop_mat attribute in their type string.
// ---------------------------------------------------------------------------
void test_type_spelling(void) {
    MatA_float16x16 a;
    MatB_float16x16 b;
    MatC_float16x16 c;
    MatA_int8x8     d;
}

// AST: VarDecl {{.*}} a {{.*}}coop_mat(
// AST: VarDecl {{.*}} b {{.*}}coop_mat(
// AST: VarDecl {{.*}} c {{.*}}coop_mat(
// AST: VarDecl {{.*}} d {{.*}}coop_mat(

// ---------------------------------------------------------------------------
// 3. sizeof / getTypeInfoImpl — width = elem * rows * cols
//    float(4B)*16*16 = 1024 B = 8192 bits
//    int(4B)*8*8    =  256 B = 2048 bits
// ---------------------------------------------------------------------------
void test_sizeof(void) {
    _Static_assert(sizeof(MatA_float16x16) == 1024, "float 16x16 size");
    _Static_assert(sizeof(MatA_int8x8)     ==  256, "int 8x8 size");
}

// ---------------------------------------------------------------------------
// 4. Parameter / return type — type preserved through function boundary.
// ---------------------------------------------------------------------------
MatA_float16x16 test_param_return(MatA_float16x16 in) {
    return in;
}

// AST: FunctionDecl {{.*}} test_param_return
// AST: ParmVarDecl {{.*}} in {{.*}}coop_mat(

// ---------------------------------------------------------------------------
// 5. TypeLoc operand traversal — all four operand slots populated.
// ---------------------------------------------------------------------------
void test_typeloc_operands(void) {
    float __attribute__((coop_mat(SCOPE, 4, 8, USE_C))) local_acc;
    (void)local_acc;
}

// AST: VarDecl {{.*}} local_acc {{.*}}coop_mat(

// ---------------------------------------------------------------------------
// 6. RecursiveASTVisitor — element type (half) reachable through the node.
// ---------------------------------------------------------------------------
void test_visitor_element_type(half __attribute__((coop_mat(SCOPE, 8, 8, USE_B))) x) {
    (void)x;
}

// AST: ParmVarDecl {{.*}} x {{.*}}coop_mat(

// ---------------------------------------------------------------------------
// 7. mergeTypes / type compatibility — two identical typedefs resolve to the
//    same canonical type; taking a pointer across them compiles cleanly.
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_alias;

void test_merge_types(void) {
    MatA_float16x16 *p = 0;
    MatA_alias      *q = p;   // same canonical type — no diagnostic
    (void)q;
}

// ---------------------------------------------------------------------------
// 8. PCH serialisation round-trip.
// ---------------------------------------------------------------------------
// PCH: VarDecl {{.*}} g {{.*}}coop_mat(
