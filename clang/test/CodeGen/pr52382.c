// RUN: %clang -target x86_64-unknown-linux-gnu -S -emit-llvm -o - -fsanitize=address %s | FileCheck %s

// Ensure that ASan properly instruments a load into a global where the index
// happens to be within the padding after the global which is used for the
// redzone.

// This global is 400 bytes long, but gets padded with 112 bytes for redzones,
// rounding the total size after instrumentation to 512.
int global_array[100] = {-1};

// This access is 412 bytes after the start of the global: past the end of the
// uninstrumented array, but within the bounds of the extended instrumented
// array. We should ensure this is still instrumented.
int main(void) { return global_array[103]; }

// CHECK: @main
// CHECK-NEXT: entry:
// CHECK: call void @__asan_report_load4
// CHECK: }
