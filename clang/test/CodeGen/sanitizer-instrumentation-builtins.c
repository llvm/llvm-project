// Check that the sanitizer region markers are available, and that they lower to
// nothing without a sanitizer that consumes them.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes \
// RUN:     -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=thread \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

#if !__has_builtin(__builtin_sanitizer_instrumentation_begin)
#error "__builtin_sanitizer_instrumentation_begin not available"
#endif
#if !__has_builtin(__builtin_sanitizer_instrumentation_end)
#error "__builtin_sanitizer_instrumentation_end not available"
#endif

extern void instrumented_callee(int);

__attribute__((disable_sanitizer_instrumentation))
void noinstr_function(int x) {
  __builtin_sanitizer_instrumentation_begin();
  instrumented_callee(x);
  __builtin_sanitizer_instrumentation_end();
}

// CHECK-LABEL: define {{.*}}void @noinstr_function(
// CHECK-NOT:     llvm.sanitizer.instrumentation
// CHECK:       }
