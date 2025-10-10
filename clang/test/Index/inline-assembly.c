static void inline_assembly_template_regardless_of_target_machine() {
    int tmp;
    asm volatile (
        "nop\n"
        "a_value %w[v]\n"
        "o_value %w[o]"
        : [v] "=&r" (tmp)
        : [o] "r" (tmp)
        : "cc", "memory"
    );
}

// RUN: c-index-test -test-inline-assembly %s 2>&1 | FileCheck %s
// CHECK: ===ASM TEMPLATE===
// CHECK: nop
// CHECK: a_value ${0:w}
// CHECK: o_value ${1:w}
// CHECK: ===ASM TEMPLATE END===
// CHECK: volatile: true
// CHECK: Output #0 Constraint (=&r): DeclRefExpr=tmp:2:9
// CHECK: Input #0 Constraint (r): UnexposedExpr=tmp:2:9
// CHECK: Clobber #0: cc
// CHECK: Clobber #1: memory
// CHECK: ===ASM END===

static void inline_assembly_valid_x86_example() {
    int tmp;
    asm (
        "nop\n"
        "mov %w[o], %w[v]"
        : [v] "=&r" (tmp)
        : [o] "r" (tmp)
        : "cc", "memory"
    );
}

// CHECK: ===ASM TEMPLATE===
// CHECK: nop
// CHECK: mov ${1:w}, ${0:w}
// CHECK: ===ASM TEMPLATE END===
// CHECK: volatile: false
// CHECK: Output #0 Constraint (=&r): DeclRefExpr=tmp:27:9
// CHECK: Input #0 Constraint (r): UnexposedExpr=tmp:27:9
// CHECK: Clobber #0: cc
// CHECK: Clobber #1: memory
// CHECK: ===ASM END===
