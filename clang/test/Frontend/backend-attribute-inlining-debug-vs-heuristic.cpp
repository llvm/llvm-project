// RUN: %clang_cc1 -O2 -emit-obj %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=HEURISTIC
// RUN: %clang_cc1 -O2 -emit-obj -debug-info-kind=line-directives-only %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// Verify auto-selection works between debug info and heuristic fallback. When
// we have at least -gline-directives-only we can use DILocation for accurate
// inline locations.

// Without that debug info we fall back to a heuristic approach using srcloc
// metadata.

[[gnu::warning("dangerous function")]]
void dangerous();

// Non-static, non-inline functions that get inlined at -O2.
void wrapper() {
    dangerous();
}

void middle() {
    wrapper();
}

void caller() {
    middle();
}


// HEURISTIC: :16:{{.*}}: warning: call to '{{.*}}dangerous{{.*}}'
// HEURISTIC: :16:{{.*}}: note: called by function '{{.*}}wrapper{{.*}}'
// HEURISTIC: :16:{{.*}}: note: inlined by function '{{.*}}middle{{.*}}'
// HEURISTIC: :16:{{.*}}: note: inlined by function '{{.*}}caller{{.*}}'
// HEURISTIC: note: use '-gline-directives-only' (implied by '-g1' or higher) for more accurate inlining chain locations

// DEBUG: :16:{{.*}}: warning: call to '{{.*}}dangerous{{.*}}'
// DEBUG: :16:{{.*}}: note: called by function '{{.*}}wrapper{{.*}}'
// DEBUG: :20:{{.*}}: note: inlined by function '{{.*}}middle{{.*}}'
// DEBUG: :24:{{.*}}: note: inlined by function '{{.*}}caller{{.*}}'
// DEBUG-NOT: note: use '-gline-directives-only'

// Test that functions in anonymous namespaces are properly tracked for
// inlining chain diagnostics. Anonymous namespace functions have internal
// linkage and are prime candidates for inlining.

[[gnu::warning("do not call")]]
void bad_func();

namespace {
void anon_helper() {
    bad_func();
}

void anon_middle() {
    anon_helper();
}
} // namespace

void public_caller() {
    anon_middle();
}

// HEURISTIC: :49:{{.*}}: warning: call to '{{.*}}bad_func{{.*}}'
// HEURISTIC: :49:{{.*}}: note: called by function '{{.*}}anon_helper{{.*}}'
// HEURISTIC: :53:{{.*}}: note: inlined by function '{{.*}}anon_middle{{.*}}'
// HEURISTIC: :58:{{.*}}: note: inlined by function '{{.*}}public_caller{{.*}}'

// DEBUG: :49:{{.*}}: warning: call to '{{.*}}bad_func{{.*}}'
// DEBUG: :49:{{.*}}: note: called by function '{{.*}}anon_helper{{.*}}'
// DEBUG: :53:{{.*}}: note: inlined by function '{{.*}}anon_middle{{.*}}'
// DEBUG: :58:{{.*}}: note: inlined by function '{{.*}}public_caller{{.*}}'

// always_inline forces inlining but doesn't imply
// isInlined() in the language sense.

[[gnu::warning("always inline warning")]]
void always_inline_target();

__attribute__((always_inline))
void always_inline_wrapper() {
    always_inline_target();
}

void always_inline_caller() {
    always_inline_wrapper();
}

// HEURISTIC: :79:{{.*}}: warning: call to '{{.*}}always_inline_target{{.*}}'
// HEURISTIC: :79:{{.*}}: note: called by function '{{.*}}always_inline_wrapper{{.*}}'
// HEURISTIC: :83:{{.*}}: note: inlined by function '{{.*}}always_inline_caller{{.*}}'

// DEBUG: :79:{{.*}}: warning: call to '{{.*}}always_inline_target{{.*}}'
// DEBUG: :79:{{.*}}: note: called by function '{{.*}}always_inline_wrapper{{.*}}'
// DEBUG: :83:{{.*}}: note: inlined by function '{{.*}}always_inline_caller{{.*}}'
