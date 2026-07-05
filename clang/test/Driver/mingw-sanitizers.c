// RUN: touch %t.a
// RUN: %clang --target=i686-windows-gnu %s -### -fsanitize=address -lcomponent %/t.a 2>&1 | FileCheck --check-prefixes=ASAN-ALL,ASAN-I686 -DINPUT=%/t.a %s
// RUN: %clang --target=x86_64-windows-gnu %s -### -fsanitize=address -lcomponent %/t.a 2>&1 | FileCheck --check-prefixes=ASAN-ALL,ASAN-X86_64 -DINPUT=%/t.a %s
//
// ASAN-ALL-NOT:"-l{{[^"]+"]}}"
// ASAN-ALL-NOT:"[[INPUT]]"
// ASAN-I686:   "{{[^"]*}}libclang_rt.asan_dynamic.dll.a"
// ASAN-X86_64: "{{[^"]*}}libclang_rt.asan_dynamic.dll.a"
// ASAN-ALL:    "-lcomponent"
// ASAN-ALL:    "[[INPUT]]"
// ASAN-I686:   "{{[^"]*}}libclang_rt.asan_dynamic.dll.a"
// ASAN-I686:   "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk.a"
// ASAN-I686:   "--require-defined" "___asan_seh_interceptor"
// ASAN-I686:   "--whole-archive" "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk.a" "--no-whole-archive"
// ASAN-X86_64: "{{[^"]*}}libclang_rt.asan_dynamic.dll.a"
// ASAN-X86_64: "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk.a"
// ASAN-X86_64: "--require-defined" "__asan_seh_interceptor"
// ASAN-X86_64: "--whole-archive" "{{[^"]*}}libclang_rt.asan_dynamic_runtime_thunk.a" "--no-whole-archive"

// RUN: %clang --target=x86_64-windows-gnu %s -### -fsanitize=vptr
