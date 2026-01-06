// RUN: %clang_cc1 -O2 -emit-obj %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED-HEURISTIC
// RUN: %clang_cc1 -O2 -emit-obj -debug-info-kind=line-directives-only %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED-DEBUG

[[gnu::warning("do not call")]]
void bad_func(void);

static inline void level1(void) {
    bad_func();
}

static inline void level2(void) {
    level1();
}

void entry(void) {
    level2();
}

// Enabled without debug info: heuristic fallback.
// All notes point to original call site (:14).
// ENABLED-HEURISTIC: :8:{{.*}}: warning: call to 'bad_func'
// ENABLED-HEURISTIC: :8:{{.*}}: note: called by function 'level1'
// ENABLED-HEURISTIC: :12:{{.*}}: note: inlined by function 'level2'
// ENABLED-HEURISTIC: :16:{{.*}}: note: inlined by function 'entry'
// ENABLED-HEURISTIC: note: use '-gline-directives-only' (implied by '-g1' or higher) for more accurate inlining chain locations

// Enabled with debug info: accurate locations.
// ENABLED-DEBUG: :8:{{.*}}: warning: call to 'bad_func'
// ENABLED-DEBUG: :8:{{.*}}: note: called by function 'level1'
// ENABLED-DEBUG: :12:{{.*}}: note: inlined by function 'level2'
// ENABLED-DEBUG: :16:{{.*}}: note: inlined by function 'entry'
// ENABLED-DEBUG-NOT: note: use '-gline-directives-only'
