// UNSUPPORTED: system-windows

// An ASan-instrumented libamdhip64.so only works if the matching ASan runtime
// is in the process and initialized before it.

// RUN: rm -rf %t && mkdir -p %t
// RUN: touch %t/in.o

// A HIP runtime without the HIP_RUNTIME_SANITIZER field.
// RUN: cp -R %S/Inputs/rocm %t/rocm

// The same runtime, built with the address sanitizer.
// RUN: cp -R %S/Inputs/rocm %t/rocm-asan
// RUN: printf 'HIP_RUNTIME_SANITIZER=address\n' >> %t/rocm-asan/bin/.hipVersion

// The ASan runtime components are linked, in the order Clang uses for a
// shared-ASan executable, and ahead of libamdhip64.so so that the ASan
// runtime precedes it in DT_NEEDED.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=ASAN %s

// ASAN:      "{{.*}}libclang_rt.asan{{[^"]*}}.so"
// ASAN-SAME: "--whole-archive" "{{.*}}libclang_rt.asan-preinit{{[^"]*}}.a" "--no-whole-archive"
// ASAN-SAME: "--whole-archive" "{{.*}}libclang_rt.asan_static{{[^"]*}}.a" "--no-whole-archive"
// ASAN-SAME: "{{.*}}in.o"
// ASAN-SAME: "{{.*}}libamdhip64.so"
// ASAN-SAME: "--no-as-needed" "-lpthread" "-lrt" "-lm" "-ldl" "-lresolv"

// Nothing is added for a HIP runtime that was not built with the sanitizer.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s

// NOASAN-NOT: libclang_rt.asan

// Nothing is added when libamdhip64.so is not linked in the first place.
// RUN: %clang -### --hip-link -nostdlib --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s
// RUN: %clang -### --hip-link -no-hip-rt --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s
// RUN: %clang -### --hip-link --no-offloadlib --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s
// RUN: %clang -### --hip-link -r --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s

// Nor for a link that does not involve HIP at all.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NOASAN %s

// An explicit '-fsanitize=address -shared-libsan' already emits the runtime,
// so it shouldn't be emitted a second time.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -fsanitize=address -shared-libsan \
// RUN:   %t/in.o 2>&1 | FileCheck -check-prefix=EXPLICIT %s

// EXPLICIT:     "{{.*}}libclang_rt.asan{{[^"]*}}.so"
// EXPLICIT-NOT: "{{.*}}libclang_rt.asan{{[^"]*}}.so"

// The HIP runtime is a shared library, so an explicit request for the static
// ASan runtime cannot be honored and is diagnosed.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -static-libsan %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=STATIC-LIBSAN %s

// STATIC-LIBSAN: warning: ignoring '-static-libsan': the HIP runtime is built with the address sanitizer and requires the shared address sanitizer runtime [-Whip-runtime-sanitizer]
// STATIC-LIBSAN: "{{.*}}libclang_rt.asan{{[^"]*}}.so"

// The warning is suppressible.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -static-libsan -Wno-hip-runtime-sanitizer \
// RUN:   %t/in.o 2>&1 | FileCheck -check-prefix=NOWARN %s

// NOWARN-NOT: warning:

// '-fsanitize=address' defaults to the static runtime, which is not compatible
// with an ASan-instrumented shared HIP runtime.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -fsanitize=address %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=STATIC-RT %s

// STATIC-RT: warning: the HIP runtime is built with the address sanitizer and requires the shared address sanitizer runtime; the static runtime selected here is not compatible with it, consider '-shared-libasan' [-Whip-runtime-sanitizer]

// The RPATH must point at the directory the selected runtime actually lives in.
// RUN: mkdir -p %t/rd/lib/linux
// RUN: touch %t/rd/lib/linux/libclang_rt.asan-x86_64.so
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -resource-dir=%t/rd -frtlib-add-rpath \
// RUN:   %t/in.o 2>&1 | FileCheck -check-prefix=RPATH %s

// RPATH: "{{.*}}/rd/lib/linux/libclang_rt.asan-x86_64.so"
// RPATH-SAME: "-rpath" "{{.*}}/rd/lib/linux"

// No RPATH without -frtlib-add-rpath.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%t/rocm-asan -resource-dir=%t/rd %t/in.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NORPATH %s

// NORPATH-NOT: "-rpath" "{{.*}}/rd/lib/linux"

//
// The user's translation units are not instrumented: this only fixes the
// loader contract, it does not imply '-fsanitize=address' for compile jobs.
//
// RUN: %clang -### -c --target=x86_64-linux-gnu --offload-arch=gfx908 \
// RUN:   -nogpuinc -nogpulib --rocm-path=%t/rocm-asan %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NOINSTR %s

// NOINSTR-NOT: "-fsanitize=address"
