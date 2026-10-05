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
// KMSAN:         call ptr @__msan_get_context_state()
// KMSAN:         call { ptr, ptr } @__msan_metadata_ptr_for_load_4(
// KASAN:         call void @__asan_report_load4(
// CHECK:         call void @instrumented_callee(
// CHECK:         call void @llvm.sanitizer.instrumentation.end()
// CHECK-NOT:     call {{.*}}@__{{msan|asan|tsan|sanitizer_cov}}
// CHECK:         ret void
