// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: not %clang -O2 -S %t/main.c -I%t -o /dev/null 2>&1 | FileCheck %s

// Cross-TU inlining: header functions inlined into source file.

//--- overflow.h
[[gnu::warning("write overflow")]]
void __write_overflow(void);

[[gnu::error("read overflow")]]
void __read_overflow(void);

static inline void check_write(int size) {
  if (size > 100)
    __write_overflow();
}

static inline void check_read(int size) {
  if (size > 50)
    __read_overflow();
}

static inline void check_both(int size) {
  check_write(size);
  check_read(size);
}

//--- main.c
#include "overflow.h"

void test_simple_cross_tu(void) {
  check_write(200);
}
// CHECK: warning: call to '__write_overflow' declared with 'warning' attribute: write overflow
// CHECK: note: called by function 'check_write'
// CHECK: main.c:{{.*}}: note: inlined by function 'test_simple_cross_tu'

// Nested cross-TU inlining (header -> header -> source).
static inline void local_wrapper(int x) {
  check_both(x);
}

void test_nested_cross_tu(void) {
  local_wrapper(200);
}
// CHECK: warning: call to '__write_overflow' declared with 'warning' attribute: write overflow
// CHECK: note: called by function 'check_write'
// CHECK: overflow.h:{{.*}}: note: inlined by function 'check_both'
// CHECK: main.c:{{.*}}: note: inlined by function 'local_wrapper'
// CHECK: main.c:{{.*}}: note: inlined by function 'test_nested_cross_tu'

// CHECK: error: call to '__read_overflow' declared with 'error' attribute: read overflow
// CHECK: note: called by function 'check_read'
// CHECK: overflow.h:{{.*}}: note: inlined by function 'check_both'
// CHECK: main.c:{{.*}}: note: inlined by function 'local_wrapper'
// CHECK: main.c:{{.*}}: note: inlined by function 'test_nested_cross_tu'

void test_error_cross_tu(void) {
  check_read(100);
}
// CHECK: error: call to '__read_overflow' declared with 'error' attribute: read overflow
// CHECK: note: called by function 'check_read'
// CHECK: main.c:{{.*}}: note: inlined by function 'test_error_cross_tu'

// Fallback note should appear (no debug info).
// CHECK: note: use '-gline-directives-only' (implied by '-g1' or higher) for more accurate inlining chain locations
