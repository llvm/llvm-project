// Check that unbalanced sanitizer instrumentation regions are diagnosed with a
// source location, and that the warning can be disabled.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-address \
// RUN:     -debug-info-kind=line-tables-only -emit-llvm -o /dev/null %s -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-memory \
// RUN:     -fsanitize-coverage-type=3 -fsanitize-coverage-trace-pc \
// RUN:     -debug-info-kind=line-tables-only -emit-llvm -o /dev/null %s -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-address \
// RUN:     -Wno-sanitizer-instrumentation-region -emit-llvm -o /dev/null %s \
// RUN:     -verify=none

// none-no-diagnostics

extern void callee(void);

__attribute__((disable_sanitizer_instrumentation))
void balanced(int c) {
  __builtin_sanitizer_instrumentation_begin();
  callee();
  __builtin_sanitizer_instrumentation_end();
  if (c) {
    __builtin_sanitizer_instrumentation_begin();
    callee();
    __builtin_sanitizer_instrumentation_end();
  }
}

// Not closing a region before a call that does not return is fine.
__attribute__((noreturn)) extern void panic(void);

__attribute__((disable_sanitizer_instrumentation))
void open_at_noreturn(int c) {
  __builtin_sanitizer_instrumentation_begin();
  if (c)
    panic();
  if (c > 1)
    __builtin_trap();
  callee();
  __builtin_sanitizer_instrumentation_end();
}

__attribute__((disable_sanitizer_instrumentation))
void unmatched_end(void) {
  callee();
  __builtin_sanitizer_instrumentation_end(); // expected-warning {{in function 'unmatched_end': sanitizer instrumentation region end without a matching begin}}
}

__attribute__((disable_sanitizer_instrumentation))
void not_closed(void) {
  __builtin_sanitizer_instrumentation_begin();
  callee();
} // expected-warning {{in function 'not_closed': sanitizer instrumentation region is not closed before return}}

__attribute__((disable_sanitizer_instrumentation))
void conditional_begin(int c) {
  if (c)
    __builtin_sanitizer_instrumentation_begin();
  callee(); // expected-warning {{in function 'conditional_begin': control flow merges from paths with different sanitizer instrumentation region nesting}}
  __builtin_sanitizer_instrumentation_end();
}
