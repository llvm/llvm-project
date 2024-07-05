// UNSUPPORTED: system-windows

// RUN: %clang -### %s -fsanitize=address --target=x86_64-pc-linux-gnu 2>&1 | FileCheck -check-prefix=CHECK-ASAN-PC %s
// CHECK-ASAN-PC: x86_64-unknown-linux-gnu/libclang_rt.asan_static.a
// RUN: %clang -### %s -fsanitize=address --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck -check-prefix=CHECK-ASAN-UNKNOWN %s
// CHECK-ASAN-UNKNOWN: x86_64-unknown-linux-gnu/libclang_rt.asan_static.a
