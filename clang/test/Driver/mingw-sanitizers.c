// RUN: %clang -target i686-windows-gnu %s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-I686 %s
// ASAN-I686: "{{.*}}libclang_rt.asan_dynamic-i386.dll.a"
// ASAN-I686: "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk-i386.a"
// ASAN-I686: "--require-defined" "___asan_seh_interceptor"
// ASAN-I686: "--whole-archive" "{{.*}}libclang_rt.asan_dynamic_runtime_thunk-i386.a" "--no-whole-archive"

// RUN: %clang -target x86_64-windows-gnu %s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-X86_64 %s
// ASAN-X86_64: "{{.*}}libclang_rt.asan_dynamic-x86_64.dll.a"
// ASAN-X86_64: "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk-x86_64.a"
// ASAN-X86_64: "--require-defined" "__asan_seh_interceptor"
// ASAN-X86_64: "--whole-archive" "{{.*}}libclang_rt.asan_dynamic_runtime_thunk-x86_64.a" "--no-whole-archive"
