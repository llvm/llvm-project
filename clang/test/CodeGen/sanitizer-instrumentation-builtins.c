// Check that the sanitizer region markers lower to their intrinsics under
// -fsanitize=kernel-memory, kernel-address, address and thread, and with
// -fsanitize-coverage, and to nothing without a sanitizer that consumes them.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-memory \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes \
// RUN:     -o - %s | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-address \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=MARKERS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=MARKERS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize-coverage-type=3 \
// RUN:     -fsanitize-coverage-trace-pc -emit-llvm -disable-llvm-passes -o - %s \
// RUN:     | FileCheck %s --check-prefix=MARKERS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=thread \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=MARKERS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=null \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NONE

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
// CHECK:         call void @llvm.sanitizer.instrumentation.begin()
// CHECK:         call void @instrumented_callee(
// CHECK:         call void @llvm.sanitizer.instrumentation.end()

// The markers are optimization barriers, so they must not be marked readnone
// or speculatable; only the default attributes plus nomerge are expected.
// CHECK: declare void @llvm.sanitizer.instrumentation.begin() #[[ATTR:[0-9]+]]
// CHECK: attributes #[[ATTR]] = { nocallback nofree nomerge nosync nounwind willreturn }

// MARKERS-LABEL: define {{.*}}void @noinstr_function(
// MARKERS:         call void @llvm.sanitizer.instrumentation.begin()
// MARKERS:         call void @llvm.sanitizer.instrumentation.end()

// NONE-LABEL: define {{.*}}void @noinstr_function(
// NONE-NOT:     llvm.sanitizer.instrumentation
// NONE:       }
