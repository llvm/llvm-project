// CooperativeMatrixType AST node and type infrastructure tests
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fdeclare-opencl-builtins -ast-dump %s \
// RUN:   | FileCheck %s --check-prefix=AST

#define SCOPE  memory_scope_sub_group
#define USE_A  CLK_COOPERATIVE_MATRIX_A
#define USE_B  CLK_COOPERATIVE_MATRIX_B
#define USE_C  CLK_COOPERATIVE_MATRIX_ACCUMULATOR

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

// Basic type construction.
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A)))  MatA_float16x16;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B)))  MatB_float16x16;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C)))  MatC_float16x16;
typedef int   __attribute__((coop_mat(SCOPE,  8,  8, USE_A)))  MatA_int8x8;
// AST: TypedefDecl {{.*}} MatA_float16x16
// AST: TypedefDecl {{.*}} MatB_float16x16
// AST: TypedefDecl {{.*}} MatC_float16x16
// AST: TypedefDecl {{.*}} MatA_int8x8

// Type printer — VarDecls carry the coop_mat attribute in their type string.
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

// Parameter / return type — type preserved through function boundary.
MatA_float16x16 test_param_return(MatA_float16x16 in) {
    return in;
}
// AST: FunctionDecl {{.*}} test_param_return
// AST: ParmVarDecl {{.*}} in {{.*}}coop_mat(

// TypeLoc operand traversal — check if all four operand slots of coop_mat
// type are populated.
void test_typeloc_operands(void) {
    float __attribute__((coop_mat(SCOPE, 4, 8, USE_C))) local_acc;
    (void)local_acc;
}
// AST: VarDecl {{.*}} local_acc {{.*}}coop_mat(4, 4, 8, 2)

// element type (half) is preserved as part of the coop_mat type.
void test_visitor_element_type(half __attribute__((coop_mat(SCOPE, 8, 8, USE_B))) x) {
    (void)x;
}
// AST: ParmVarDecl {{.*}} x {{.*}}half{{.*}}coop_mat
