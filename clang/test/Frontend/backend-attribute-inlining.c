// RUN: not %clang -O2 -S %s -o /dev/null 2>&1 | FileCheck %s

// Single-level inlining with warning attribute.
[[gnu::warning("do not call directly")]]
void __warn_single(void);

static inline void warn_wrapper(void) {
  __warn_single();
}

void test_single_level(void) {
  warn_wrapper();
}
// CHECK: warning: call to '__warn_single' declared with 'warning' attribute: do not call directly
// CHECK: note: called by function 'warn_wrapper'
// CHECK: :12:{{.*}}: note: inlined by function 'test_single_level'

// Error attribute with inlining.
[[gnu::error("never call this")]]
void __error_func(void);

static inline void error_wrapper(void) {
  __error_func();
}

void test_error_inlined(void) {
  error_wrapper();
}
// CHECK: error: call to '__error_func' declared with 'error' attribute: never call this
// CHECK: note: called by function 'error_wrapper'
// CHECK: :27:{{.*}}: note: inlined by function 'test_error_inlined'

// Deep nesting (5 levels).
[[gnu::warning("deep call")]]
void __warn_deep(void);

static inline void deep1(void) { __warn_deep(); }
static inline void deep2(void) { deep1(); }
static inline void deep3(void) { deep2(); }
static inline void deep4(void) { deep3(); }
static inline void deep5(void) { deep4(); }

void test_deep_nesting(void) {
  deep5();
}
// CHECK: warning: call to '__warn_deep' declared with 'warning' attribute: deep call
// CHECK: note: called by function 'deep1'
// CHECK: :38:{{.*}}: note: inlined by function 'deep2'
// CHECK: :39:{{.*}}: note: inlined by function 'deep3'
// CHECK: :40:{{.*}}: note: inlined by function 'deep4'
// CHECK: :41:{{.*}}: note: inlined by function 'deep5'
// CHECK: :44:{{.*}}: note: inlined by function 'test_deep_nesting'

// Multiple call sites produce distinct diagnostics.
[[gnu::warning("deprecated")]]
void __warn_multi(void);

static inline void multi_wrapper(void) {
  __warn_multi();
}

void call_site_a(void) { multi_wrapper(); }
void call_site_b(void) { multi_wrapper(); }
void call_site_c(void) { multi_wrapper(); }

// CHECK: warning: call to '__warn_multi' declared with 'warning' attribute: deprecated
// CHECK: note: called by function 'multi_wrapper'
// CHECK: :62:{{.*}}: note: inlined by function 'call_site_a'

// CHECK: warning: call to '__warn_multi' declared with 'warning' attribute: deprecated
// CHECK: note: called by function 'multi_wrapper'
// CHECK: :63:{{.*}}: note: inlined by function 'call_site_b'

// CHECK: warning: call to '__warn_multi' declared with 'warning' attribute: deprecated
// CHECK: note: called by function 'multi_wrapper'
// CHECK: :64:{{.*}}: note: inlined by function 'call_site_c'

// Different nesting depths from same inner function.
[[gnu::warning("mixed depth")]]
void __warn_mixed(void);

static inline void mixed_inner(void) { __warn_mixed(); }
static inline void mixed_middle(void) { mixed_inner(); }

void shallow(void) { mixed_inner(); }
void deep(void) { mixed_middle(); }

// CHECK: warning: call to '__warn_mixed' declared with 'warning' attribute: mixed depth
// CHECK: note: called by function 'mixed_inner'
// CHECK: :85:{{.*}}: note: inlined by function 'shallow'

// CHECK: warning: call to '__warn_mixed' declared with 'warning' attribute: mixed depth
// CHECK: note: called by function 'mixed_inner'
// CHECK: :83:{{.*}}: note: inlined by function 'mixed_middle'
// CHECK: :86:{{.*}}: note: inlined by function 'deep'

// Incidental inlining (function not marked inline/static).
// The "inlined by" note has no location since heuristic mode doesn't track it.
[[gnu::warning("incidental")]]
void __warn_incidental(void);

void not_marked_inline(void) { __warn_incidental(); }

void test_incidental(void) { not_marked_inline(); }

// CHECK: warning: call to '__warn_incidental' declared with 'warning' attribute: incidental
// CHECK: note: called by function 'not_marked_inline'
// CHECK: note: inlined by function 'test_incidental'
// CHECK-NOT: :{{.*}}: note: inlined by function 'test_incidental'

// Fallback note should appear (no debug info).
// CHECK: note: use '-gline-directives-only' (implied by '-g1' or higher) for more accurate inlining chain locations
