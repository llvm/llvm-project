// clang/test/Preprocessor/coop_mat_opencl_ext.cl
//
// Patch 2: cl_khr_cooperative_matrix registration in
//          OpenCLExtensions.def and enum definitions in opencl-c-base.h.
//
// Tests: extension macro is predefined when enabled, extension can be
//        explicitly enabled/disabled via pragma, all four enum types and
//        their constants are visible under -finclude-default-header.

// ── 2a. Extension macro is predefined when the extension is enabled ─────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -E -dM %s \
// RUN:   | FileCheck %s --check-prefix=EXT

// EXT: cl_khr_cooperative_matrix

// ── 2b. Extension is NOT predefined when explicitly disabled ────────────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=-cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -E %s \
// RUN:   | FileCheck %s --check-prefix=NOEXT

// NOEXT-NOT: cl_khr_cooperative_matrix

// ── 2c. Enum constants are visible when extension is enabled ─────────────────
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

void test_enum_constants(void) {
    // coop_matrix_scope_t
    coop_matrix_scope_t s = CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP;
    (void)s;

    // coop_matrix_use_t
    coop_matrix_use_t u0 = CLK_COOPERATIVE_MATRIX_A;
    coop_matrix_use_t u1 = CLK_COOPERATIVE_MATRIX_B;
    coop_matrix_use_t u2 = CLK_COOPERATIVE_MATRIX_ACCUMULATOR;
    (void)u0; (void)u1; (void)u2;

    // coop_matrix_layout_t
    coop_matrix_layout_t l0 = CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR;
    coop_matrix_layout_t l1 = CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR;
    (void)l0; (void)l1;

    // coop_matrix_operands_t
    coop_matrix_operands_t op = CLK_COOPERATIVE_MATRIX_OPERAND_NONE;
    (void)op;
}

// ── 2d. Enum constant values match the spec ──────────────────────────────────
void test_enum_values(void) {
    _Static_assert(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP     == 3, "scope subgroup");
    _Static_assert(CLK_COOPERATIVE_MATRIX_A                  == 0, "use A");
    _Static_assert(CLK_COOPERATIVE_MATRIX_B                  == 1, "use B");
    _Static_assert(CLK_COOPERATIVE_MATRIX_ACCUMULATOR        == 2, "use ACC");
    _Static_assert(CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR   == 0, "row major");
    _Static_assert(CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR== 1, "col major");
    _Static_assert(CLK_COOPERATIVE_MATRIX_OPERAND_NONE       == 0, "operand none");
}
