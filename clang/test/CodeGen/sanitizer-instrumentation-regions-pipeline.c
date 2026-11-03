// Check that, after the full -O2 pipeline, each sanitizer instruments the code
// inside a region of a function it otherwise does not instrument, and nothing
// else. The region markers come from an always_inline helper, as in the Linux
// kernel. Debug info is enabled so that the verifier checks that the
// instrumentation calls get debug locations.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -debug-info-kind=line-tables-only \
// RUN:     -fsanitize=kernel-memory -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,KMSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -debug-info-kind=line-tables-only \
// RUN:     -fsanitize=kernel-address -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,KASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -debug-info-kind=line-tables-only \
// RUN:     -fsanitize-coverage-type=3 -fsanitize-coverage-trace-pc \
// RUN:     -fsanitize-coverage-trace-cmp -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,KCOV
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -debug-info-kind=line-tables-only \
// RUN:     -fsanitize=kernel-memory -fsanitize-coverage-type=3 \
// RUN:     -fsanitize-coverage-trace-pc -fsanitize-coverage-trace-cmp \
// RUN:     -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,KMSAN,KCOV
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -debug-info-kind=line-tables-only \
// RUN:     -fsanitize=thread -mllvm -tsan-instrument-func-entry-exit=0 \
// RUN:     -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KCSAN

static inline __attribute__((always_inline)) void instrumentation_begin(void) {
  __builtin_sanitizer_instrumentation_begin();
}

static inline __attribute__((always_inline)) void instrumentation_end(void) {
  __builtin_sanitizer_instrumentation_end();
}

void instrumented_callee(int);

__attribute__((noinline, disable_sanitizer_instrumentation))
void noinstr_function(int *p, int x) {
  int v = *p + (x == 42);
  instrumentation_begin();
  if (x == 7)
    instrumented_callee(*p + v);
  instrumentation_end();
  *p = v;
}

// CHECK-LABEL: define {{.*}}void @noinstr_function(
// CHECK-NOT:     call {{.*}}@__{{msan|asan|tsan|sanitizer_cov}}
// CHECK:         call void @llvm.sanitizer.instrumentation.begin()
// KCOV:          call void @__sanitizer_cov_trace_pc()
// KCOV:          call void @__sanitizer_cov_trace_const_cmp4(i32 7,
// KMSAN:         call { ptr, ptr } @__msan_metadata_ptr_for_load_4(
// KASAN:         call void @__asan_report_load4(
// KCSAN:         call void @__tsan_read4(
// The argument is checked eagerly (-fsanitize-memory-param-retval), so the
// region never touches the context state and does not fetch it.
// KMSAN:         call void @__msan_warning(
// KMSAN-NOT:     @__msan_get_context_state
// CHECK:         call void @instrumented_callee(
// CHECK:         call void @llvm.sanitizer.instrumentation.end()
// CHECK-NOT:     call {{.*}}@__{{msan|asan|tsan|sanitizer_cov}}
// CHECK:         ret void
