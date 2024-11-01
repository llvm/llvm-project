// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s | FileCheck -check-prefixes CHECK,WITHOUT %s
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s -fsanitize=memory | FileCheck -check-prefixes CHECK,MSAN %s
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s -fsanitize=kernel-memory | FileCheck -check-prefixes CHECK,KMSAN %s
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s -fsanitize=memory -fno-sanitize-memory-param-retval | FileCheck -check-prefixes CHECK,MSAN,RETTLS %s

// Instrumented function.
// MSan uses memset(addr, -1, size) to poison allocas and stores shadow of the return value in
// __msan_retval_tls. KMSAN uses __msan_poison_alloca() to poison allocas and calls
// __msan_get_context_state() at function prologue to access the task context struct (including the
// shadow of the return value).
//
// CHECK-LABEL: i32 @instrumented1
// KMSAN: __msan_get_context_state
// WITHOUT-NOT: __msan_poison_alloca
// WITHOUT-NOT: @llvm.memset
// MSAN: @llvm.memset{{.*}}({{.*}}, i8 -1
// KMSAN: __msan_poison_alloca
// WITHOUT-NOT: __msan_retval_tls
// RETTLS: __msan_retval_tls
// CHECK: ret i32
int instrumented1(int *a) {
  volatile char buf[8];
  return *a;
}

// Function with no_sanitize("memory")/no_sanitize("kernel-memory"): no shadow propagation, but
// unpoisons memory to prevent false positives.
// MSan uses memset(addr, 0, size) to unpoison locals, KMSAN uses __msan_unpoison_alloca(). Both
// tools still access the retval shadow to write 0 to it.
//
// CHECK-LABEL: i32 @no_false_positives1
// KMSAN: __msan_get_context_state
// WITHOUT-NOT: __msan_unpoison_alloca
// WITHOUT-NOT: @llvm.memset
// MSAN: @llvm.memset{{.*}}({{.*}}, i8 0
// KMSAN: __msan_unpoison_alloca
// WITHOUT-NOT: __msan_retval_tls
// RETTLS: __msan_retval_tls
// CHECK: ret i32
__attribute__((no_sanitize("memory"))) __attribute__((no_sanitize("kernel-memory"))) int no_false_positives1(int *a) {
  volatile char buf[8];
  return *a;
}

// Function with disable_sanitizer_instrumentation: no instrumentation at all.
//
// CHECK-LABEL: i32 @no_instrumentation1
// KMSAN-NOT: __msan_get_context_state
// WITHOUT-NOT: __msan_poison_alloca
// WITHOUT-NOT: @llvm.memset
// MSAN-NOT: @llvm.memset{{.*}}({{.*}}, i8 0
// KMSAN-NOT: __msan_unpoison_alloca
// WITHOUT-NOT: __msan_retval_tls
// MSAN-NOT: __msan_retval_tls
// CHECK: ret i32
__attribute__((disable_sanitizer_instrumentation)) int no_instrumentation1(int *a) {
  volatile char buf[8];
  return *a;
}
