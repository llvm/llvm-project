// REQUIRES: arm-registered-target
//
// End-to-end test: compile C to ARM assembly and verify that
// -ffp-exception-behavior=maytrap prevents if-conversion of FP operations,
// while the default (no flag) still allows if-conversion as before.
//
// The function "pick" has cheap FP ops in both branches, which is the classic
// pattern that triggers ARM if-conversion into a branchless predicated block.
//
// With maytrap: FP ops are constrained (have side-effects on FPSCR), so the
// optimizer must preserve the branch. Only the taken path executes its FP op.
//
// Without maytrap: the optimizer if-converts both paths into a single block
// with conditional moves, speculatively executing all FP ops.

// RUN: %clang -target armv7a-none-eabi -mcpu=cortex-a9 -mfloat-abi=hard -O2 \
// RUN:     -ffp-exception-behavior=maytrap -S -o - %s \
// RUN:     | FileCheck -check-prefix=MAYTRAP %s

// RUN: %clang -target armv7a-none-eabi -mcpu=cortex-a9 -mfloat-abi=hard -O2 \
// RUN:     -S -o - %s \
// RUN:     | FileCheck -check-prefix=DEFAULT %s

// --- maytrap: branch preserved, FP ops in separate basic blocks ---
// MAYTRAP-LABEL: pick:
// MAYTRAP:       beq
// MAYTRAP:       vadd.f32
// MAYTRAP:       vsub.f32

// --- default: if-converted, no branch, branchless predicated block ---
// DEFAULT-LABEL: pick:
// DEFAULT-NOT:   beq
// DEFAULT:       vadd.f32
// DEFAULT-NOT:   vsub.f32

float pick(int flag, float a, float b) {
    if (flag)
        return a + b;
    else
        return a - b;
}
