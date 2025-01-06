// Test sanitizers ld flags.

// Match all sanitizer related libclang_rt, we are not interested in
// libclang_rt.builtins, libclang_rt.osx, libclang_rt.ios, libclang_rt.watchos
// etc.
//
// If we need to add sanitizer with name starting with excluded laters, e.g.
// `bsan`, we can extend expression like this: `([^iow]|b[^u])`.
//
// DEFINE: %{filecheck} = FileCheck %s --implicit-check-not="libclang_rt.{{([^biow])}}"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX
//
// CHECK-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-NOT: "-lc"
// CHECK-ASAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-NOT: "--export-dynamic"
// CHECK-ASAN-LINUX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-LINUX-NOT: "--export-dynamic"
// CHECK-ASAN-LINUX: "-lpthread"
// CHECK-ASAN-LINUX: "-lrt"
// CHECK-ASAN-LINUX: "-ldl"
// CHECK-ASAN-LINUX: "-lresolv"

// RUN: %clang -fsanitize=address -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-NO-LINK-RUNTIME-LINUX
//
// CHECK-ASAN-NO-LINK-RUNTIME-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=address -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=arm64e-apple-macosx -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-NO-LINK-RUNTIME-DARWIN
//
// CHECK-ASAN-NO-LINK-RUNTIME-DARWIN: "{{.*}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=address -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-EXECUTABLE-LINUX
//
// CHECK-ASAN-EXECUTABLE-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-EXECUTABLE-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-EXECUTABLE-LINUX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"

// RUN: %clang -fsanitize=address -shared -### %s 2>&1  \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-SHARED-LINUX
//
// CHECK-ASAN-SHARED-LINUX: libclang_rt.asan_static

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHARED-ASAN-LINUX

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHARED-ASAN-LINUX

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=address \
// RUN:     -shared-libsan -static-libsan -shared-libasan             \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHARED-ASAN-LINUX
//
// CHECK-SHARED-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lc"
// CHECK-SHARED-ASAN-LINUX: libclang_rt.asan.so"
// CHECK-SHARED-ASAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan-preinit.a" "--no-whole-archive"
// CHECK-SHARED-ASAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lpthread"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lrt"
// CHECK-SHARED-ASAN-LINUX-NOT: "-ldl"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lresolv"
// CHECK-SHARED-ASAN-LINUX-NOT: "--export-dynamic"
// CHECK-SHARED-ASAN-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s -o %t.so -shared 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-DSO-SHARED-ASAN-LINUX
//
// CHECK-DSO-SHARED-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lc"
// CHECK-DSO-SHARED-ASAN-LINUX: libclang_rt.asan.so"
// CHECK-DSO-SHARED-ASAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lpthread"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lrt"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-ldl"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lresolv"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "--export-dynamic"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-freebsd -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-FREEBSD
//
// CHECK-ASAN-FREEBSD: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-FREEBSD-NOT: "-lc"
// CHECK-ASAN-FREEBSD: freebsd{{/|\\+}}libclang_rt.asan_static.a"
// CHECK-ASAN-FREEBSD: freebsd{{/|\\+}}libclang_rt.asan.a"
// CHECK-ASAN-FREEBSD-NOT: "--dynamic-list"
// CHECK-ASAN-FREEBSD: "--export-dynamic"
// CHECK-ASAN-FREEBSD: "-lpthread"
// CHECK-ASAN-FREEBSD: "-lrt"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-freebsd -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-FREEBSD-LDL
//
// CHECK-ASAN-FREEBSD-LDL: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-FREEBSD-LDL-NOT: "-ldl"
// CHECK-ASAN-FREEBSD-LDL: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-FREEBSD-LDL: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-CXX

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -fsanitize-link-c++-runtime \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-CXX

// CHECK-ASAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-CXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-CXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan_cxx.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-CXX-NOT: "--dynamic-list"
// CHECK-ASAN-LINUX-CXX-SAME: "--export-dynamic"
// CHECK-ASAN-LINUX-CXX-SAME: "-lstdc++"
// CHECK-ASAN-LINUX-CXX-SAME: "-lpthread"
// CHECK-ASAN-LINUX-CXX-SAME: "-lrt"
// CHECK-ASAN-LINUX-CXX-SAME: "-ldl"
// CHECK-ASAN-LINUX-CXX-SAME: "-lresolv"
// CHECK-ASAN-LINUX-CXX-SAME: "-lc"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -fno-sanitize-link-c++-runtime \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-CNOCXX

// CHECK-ASAN-LINUX-CNOCXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "--export-dynamic"
// CHECK-ASAN-LINUX-CNOCXX-NOT: stdc++
// CHECK-ASAN-LINUX-CNOCXX-SAME: "-lpthread"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "-lrt"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "-ldl"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "-lresolv"
// CHECK-ASAN-LINUX-CNOCXX-SAME: "-lc"

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -fno-sanitize-link-c++-runtime \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-NOCXX

// CHECK-ASAN-LINUX-NOCXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-NOCXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-NOCXX-SAME: "--export-dynamic"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-lstdc++"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-lpthread"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-lrt"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-ldl"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-lresolv"
// CHECK-ASAN-LINUX-NOCXX-SAME: "-lc"

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -nostdlib++ \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-NOSTDCXX

// CHECK-ASAN-LINUX-NOSTDCXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan_cxx.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "--export-dynamic"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "-lpthread"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "-lrt"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "-ldl"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "-lresolv"
// CHECK-ASAN-LINUX-NOSTDCXX-SAME: "-lc"

// RUN: %clang -### %s -o /dev/null -fsanitize=address \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree -lstdc++ -static 2>&1 \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-LINUX-CXX-STATIC
//
// CHECK-ASAN-LINUX-CXX-STATIC: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-CXX-STATIC-NOT: stdc++
// CHECK-ASAN-LINUX-CXX-STATIC: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-LINUX-CXX-STATIC: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-LINUX-CXX-STATIC: stdc++

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-gnueabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ARM
//
// CHECK-ASAN-ARM: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ARM-NOT: "-lc"
// CHECK-ASAN-ARM: libclang_rt.asan_static.a"
// CHECK-ASAN-ARM: libclang_rt.asan.a"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=armv7l-linux-gnueabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ARMv7
//
// CHECK-ASAN-ARMv7: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ARMv7-NOT: "-lc"
// CHECK-ASAN-ARMv7: libclang_rt.asan_static.a"
// CHECK-ASAN-ARMv7: libclang_rt.asan.a"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID
//
// CHECK-ASAN-ANDROID: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-ASAN-ANDROID: "-pie"
// CHECK-ASAN-ANDROID-NOT: "-lc"
// CHECK-ASAN-ANDROID-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-NOT: "-lresolv"
// CHECK-ASAN-ANDROID: libclang_rt.asan.so"
// CHECK-ASAN-ANDROID: libclang_rt.asan_static.a"
// CHECK-ASAN-ANDROID-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-NOT: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -static-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID-STATICLIBASAN
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -static-libasan \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID-STATICLIBASAN
//
// CHECK-ASAN-ANDROID-STATICLIBASAN: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-STATICLIBASAN: libclang_rt.asan_static.a"
// CHECK-ASAN-ANDROID-STATICLIBASAN: libclang_rt.asan.a"
// CHECK-ASAN-ANDROID-STATICLIBASAN-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-STATICLIBASAN-NOT: "-lrt"
// CHECK-ASAN-ANDROID-STATICLIBASAN-NOT: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=undefined \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-ANDROID
//
// CHECK-UBSAN-ANDROID: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-UBSAN-ANDROID: "-pie"
// CHECK-UBSAN-ANDROID-NOT: "-lc"
// CHECK-UBSAN-ANDROID-NOT: "-lpthread"
// CHECK-UBSAN-ANDROID-NOT: "-lresolv"
// CHECK-UBSAN-ANDROID: libclang_rt.ubsan_standalone.so"
// CHECK-UBSAN-ANDROID-NOT: "-lpthread"
// CHECK-UBSAN-ANDROID-NOT: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=undefined \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -static-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-ANDROID-STATICLIBASAN
//
// CHECK-UBSAN-ANDROID-STATICLIBASAN: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-UBSAN-ANDROID-STATICLIBASAN: libclang_rt.ubsan_standalone.a"
// CHECK-UBSAN-ANDROID-STATICLIBASAN-NOT: "-lpthread"
// CHECK-UBSAN-ANDROID-STATICLIBASAN-NOT: "-lrt"
// CHECK-UBSAN-ANDROID-STATICLIBASAN-NOT: "-lresolv"

//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i686-linux-android -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID-X86
//
// CHECK-ASAN-ANDROID-X86: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-X86: "-pie"
// CHECK-ASAN-ANDROID-X86-NOT: "-lc"
// CHECK-ASAN-ANDROID-X86-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-X86-NOT: "-lresolv"
// CHECK-ASAN-ANDROID-X86: libclang_rt.asan.so"
// CHECK-ASAN-ANDROID-X86: libclang_rt.asan_static.a"
// CHECK-ASAN-ANDROID-X86-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-X86-NOT: "-lresolv"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID-SHARED-LIBASAN
//
// CHECK-ASAN-ANDROID-SHARED-LIBASAN-NOT: argument unused during compilation: '-shared-libsan'
// CHECK-ASAN-ANDROID-SHARED-LIBASAN: libclang_rt.asan{{.*}}.so"
// CHECK-ASAN-ANDROID-SHARED-LIBASAN: libclang_rt.asan_static{{.*}}.a"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-ANDROID-SHARED
//
// CHECK-ASAN-ANDROID-SHARED: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-SHARED-NOT: "-lc"
// CHECK-ASAN-ANDROID-SHARED: libclang_rt.asan.so"
// CHECK-ASAN-ANDROID-SHARED: libclang_rt.asan_static.a"
// CHECK-ASAN-ANDROID-SHARED-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-SHARED-NOT: "-lresolv"


// RUN: %clangxx %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -stdlib=platform -lstdc++ \
// RUN:     -fsanitize=type \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TYSAN-LINUX-CXX
//
// CHECK-TYSAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-TYSAN-LINUX-CXX-NOT: stdc++
// CHECK-TYSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.tysan{{[^.]*}}.a" "--no-whole-archive"
// CHECK-TYSAN-LINUX-CXX: stdc++

// RUN: %clangxx -fsanitize=type -### %s 2>&1 \
// RUN:     -mmacosx-version-min=10.6 \
// RUN:     --target=x86_64-apple-darwin13.4.0 -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TYSAN-DARWIN-CXX
// CHECK-TYSAN-DARWIN-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-TYSAN-DARWIN-CXX: libclang_rt.tysan_osx_dynamic.dylib
// CHECK-TYSAN-DARWIN-CXX-NOT: -lc++abi

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -stdlib=platform -lstdc++ \
// RUN:     -fsanitize=thread \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TSAN-LINUX-CXX
//
// CHECK-TSAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-TSAN-LINUX-CXX-NOT: stdc++
// CHECK-TSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.tsan.a" "--no-whole-archive"
// CHECK-TSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan.a.syms"
// CHECK-TSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.tsan_cxx.a" "--no-whole-archive"
// CHECK-TSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan_cxx.a.syms"
// CHECK-TSAN-LINUX-CXX-NOT: "--export-dynamic"
// CHECK-TSAN-LINUX-CXX: stdc++
// CHECK-TSAN-LINUX-CXX: "-lpthread"
// CHECK-TSAN-LINUX-CXX: "-lrt"
// CHECK-TSAN-LINUX-CXX: "-ldl"
// CHECK-TSAN-LINUX-CXX: "-lresolv"

// RUN: %clang -fsanitize=thread -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TSAN-NO-LINK-RUNTIME-LINUX
//
// CHECK-TSAN-NO-LINK-RUNTIME-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: not %clang -fsanitize=thread -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=arm64e-apple-ios -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TSAN-NO-LINK-RUNTIME-DARWIN
//
// CHECK-TSAN-NO-LINK-RUNTIME-DARWIN: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -stdlib=platform -lstdc++ \
// RUN:     -fsanitize=memory \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-MSAN-LINUX-CXX
//
// CHECK-MSAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-MSAN-LINUX-CXX-NOT: stdc++
// CHECK-MSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.msan.a" "--no-whole-archive"
// CHECK-MSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan.a.syms"
// CHECK-MSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.msan_cxx.a" "--no-whole-archive"
// CHECK-MSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan_cxx.a.syms"
// CHECK-MSAN-LINUX-CXX-NOT: "--export-dynamic"
// CHECK-MSAN-LINUX-CXX: stdc++
// CHECK-MSAN-LINUX-CXX: "-lpthread"
// CHECK-MSAN-LINUX-CXX: "-lrt"
// CHECK-MSAN-LINUX-CXX: "-ldl"
// CHECK-MSAN-LINUX-CXX: "-lresolv"

// RUN: %clang -fsanitize=memory -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-MSAN-NO-LINK-RUNTIME-LINUX
//
// CHECK-MSAN-NO-LINK-RUNTIME-LINUX: "{{.*}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX

// RUN: %clang -fsanitize=float-divide-by-zero -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:     -static-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX

// CHECK-UBSAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone.a" "--no-whole-archive"
// CHECK-UBSAN-LINUX-NOT: "-lstdc++"
// CHECK-UBSAN-LINUX: "-lpthread"
// CHECK-UBSAN-LINUX: "-lresolv"

// RUN: %clang -fsanitize=undefined -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-NO-LINK-RUNTIME-LINUX
//
// CHECK-UBSAN-NO-LINK-RUNTIME-LINUX: "{{.*}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=undefined -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-NO-LINK-RUNTIME-DARWIN
//
// CHECK-UBSAN-NO-LINK-RUNTIME-DARWIN: "{{.*}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=fuzzer -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=arm64e-apple-watchos -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-FUZZER-NO-LINK-RUNTIME-DARWIN
//
// CHECK-FUZZER-NO-LINK-RUNTIME-DARWIN: "{{.*}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -static-libsan -shared-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared -shared-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN

// CHECK-UBSAN-LINUX-SHAREDLIBASAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-SHAREDLIBASAN: "{{.*}}libclang_rt.ubsan_standalone.so{{.*}}"

// RUN: %clang -fsanitize=undefined -fsanitize-link-c++-runtime -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-LINK-CXX
// CHECK-UBSAN-LINUX-LINK-CXX-NOT: "-lstdc++"
// CHECK-UBSAN-LINUX-LINK-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx.a" "--no-whole-archive"
// CHECK-UBSAN-LINUX-LINK-CXX-NOT: "-lstdc++"

// RUN: %clangxx -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-CXX
// CHECK-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone.a" "--no-whole-archive"
// CHECK-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx.a" "--no-whole-archive"
// CHECK-UBSAN-LINUX-CXX: "-lstdc++"
// CHECK-UBSAN-LINUX-CXX: "-lpthread"
// CHECK-UBSAN-LINUX-CXX: "-lresolv"

// RUN: %clang -fsanitize=undefined -fsanitize-minimal-runtime -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-MINIMAL-LINUX
// CHECK-UBSAN-MINIMAL-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-MINIMAL-LINUX: "--whole-archive" "{{.*}}libclang_rt.ubsan_minimal.a" "--no-whole-archive"
// CHECK-UBSAN-MINIMAL-LINUX: "-lpthread"
// CHECK-UBSAN-MINIMAL-LINUX: "-lresolv"

// RUN: %clang -fsanitize=undefined -fsanitize-minimal-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-MINIMAL-DARWIN
// CHECK-UBSAN-MINIMAL-DARWIN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-MINIMAL-DARWIN: "{{.*}}libclang_rt.ubsan_minimal_osx_dynamic.dylib"

// RUN: not %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld -static-libsan \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-STATIC-DARWIN
// CHECK-UBSAN-STATIC-DARWIN: {{.*}}error: static UndefinedBehaviorSanitizer runtime is not supported on darwin

// RUN: not %clang -fsanitize=address -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld -static-libsan \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-STATIC-DARWIN
// CHECK-ASAN-STATIC-DARWIN: {{.*}}error: static AddressSanitizer runtime is not supported on darwin

// RUN: not %clang -fsanitize=thread -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld -static-libsan \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TSAN-STATIC-DARWIN
// CHECK-TSAN-STATIC-DARWIN: {{.*}}error: static ThreadSanitizer runtime is not supported on darwin

// RUN: %clang -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-UBSAN-LINUX
// CHECK-ASAN-UBSAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-UBSAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-UBSAN-LINUX-NOT: "-lstdc++"
// CHECK-ASAN-UBSAN-LINUX: "-lpthread"
// CHECK-ASAN-UBSAN-LINUX: "-lresolv"

// RUN: %clangxx -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-UBSAN-LINUX-CXX
// CHECK-ASAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.asan_cxx.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-lstdc++"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-lpthread"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-lresolv"

// RUN: %clangxx -fsanitize=address,undefined -fno-sanitize=vptr -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan_static.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "--whole-archive" "{{.*}}libclang_rt.asan_cxx.a" "--no-whole-archive"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "-lstdc++"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "-lpthread"
// CHECK-ASAN-UBSAN-NOVPTR-LINUX-CXX-SAME: "-lresolv"

// RUN: %clangxx -fsanitize=memory,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-MSAN-UBSAN-LINUX-CXX
// CHECK-MSAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-MSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.msan.a" "--no-whole-archive"
// CHECK-MSAN-UBSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan.a.syms"
// CHECK-MSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.msan_cxx.a" "--no-whole-archive"
// CHECK-MSAN-UBSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan_cxx.a.syms"
// CHECK-MSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx.a" "--no-whole-archive"

// RUN: %clangxx -fsanitize=thread,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-TSAN-UBSAN-LINUX-CXX
// CHECK-TSAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-TSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.tsan.a" "--no-whole-archive"
// CHECK-TSAN-UBSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan.a.syms"
// CHECK-TSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.tsan_cxx.a" "--no-whole-archive"
// CHECK-TSAN-UBSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan_cxx.a.syms"
// CHECK-TSAN-UBSAN-LINUX-CXX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx.a" "--no-whole-archive"

// RUN: %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-LINUX-SHARED
// CHECK-UBSAN-LINUX-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-SHARED-NOT: --export-dynamic
// CHECK-UBSAN-LINUX-SHARED-NOT: --dynamic-list

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=leak \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-LSAN-LINUX
//
// CHECK-LSAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LSAN-LINUX-NOT: "-lc"
// CHECK-LSAN-LINUX: libclang_rt.lsan.a"
// CHECK-LSAN-LINUX: "-lpthread"
// CHECK-LSAN-LINUX: "-ldl"
// CHECK-LSAN-LINUX: "-lresolv"

// RUN: %clang -fsanitize=leak -fno-sanitize-link-runtime -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-LSAN-NO-LINK-RUNTIME-LINUX
//
// CHECK-LSAN-NO-LINK-RUNTIME-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: %clang -### %s 2>&1 \
// RUN:  --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=leak -fsanitize-coverage=func \
// RUN:  -resource-dir=%S/Inputs/resource_dir \
// RUN:  --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-LSAN-COV-LINUX
//
// CHECK-LSAN-COV-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LSAN-COV-LINUX-NOT: "-lc"
// CHECK-LSAN-COV-LINUX: libclang_rt.lsan.a
// CHECK-LSAV-COV-LINUX: libclang_rt.lsan-x86_64.a"
// CHECK-LSAN-COV-LINUX: "-lpthread"
// CHECK-LSAN-COV-LINUX: "-ldl"
// CHECK-LSAN-COV-LINUX: "-lresolv"

// RUN: %clang -fsanitize=leak,address -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-LSAN-ASAN-LINUX
// CHECK-LSAN-ASAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LSAN-ASAN-LINUX: libclang_rt.asan_static
// CHECK-LSAN-ASAN-LINUX: libclang_rt.asan
// CHECK-LSAN-ASAN-LINUX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"

// RUN: %clang -fsanitize=address -fsanitize-coverage=func -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-COV-LINUX
// CHECK-ASAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-COV-LINUX: libclang_rt.asan_static
// CHECK-ASAN-COV-LINUX: libclang_rt.asan
// CHECK-ASAN-COV-LINUX: "--dynamic-list={{.*}}libclang_rt.asan.a.syms"
// CHECK-ASAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-ASAN-COV-LINUX: "-lpthread"
// CHECK-ASAN-COV-LINUX: "-lresolv"

// RUN: %clang -fsanitize=memory -fsanitize-coverage=func -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-MSAN-COV-LINUX
// CHECK-MSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-MSAN-COV-LINUX: "--whole-archive" "{{.*}}libclang_rt.msan.a" "--no-whole-archive"
// CHECK-MSAN-COV-LINUX: "--dynamic-list={{.*}}libclang_rt.msan.a.syms"
// CHECK-MSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-MSAN-COV-LINUX: "-lpthread"
// CHECK-MSAN-COV-LINUX: "-lresolv"

// RUN: %clang -fsanitize=dataflow -fsanitize-coverage=func -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-DFSAN-COV-LINUX
// CHECK-DFSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-DFSAN-COV-LINUX: "--whole-archive" "{{.*}}libclang_rt.dfsan.a" "--no-whole-archive"
// CHECK-DFSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-DFSAN-COV-LINUX: "-lpthread"
// CHECK-DFSAN-COV-LINUX: "-lresolv"

// RUN: %clang -fsanitize=undefined -fsanitize-coverage=func -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-COV-LINUX
// CHECK-UBSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-COV-LINUX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone.a" "--no-whole-archive"
// CHECK-UBSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-UBSAN-COV-LINUX: "-lpthread"
// CHECK-UBSAN-COV-LINUX: "-lresolv"

// RUN: %clang -fsanitize-coverage=func -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-COV-LINUX
// CHECK-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-COV-LINUX: "--whole-archive" "{{.*}}libclang_rt.ubsan_standalone.a" "--no-whole-archive"
// CHECK-COV-LINUX-NOT: "-lstdc++"
// CHECK-COV-LINUX: "-lpthread"
// CHECK-COV-LINUX: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=numerical \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-NSAN-LINUX
//
// CHECK-NSAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-NSAN-LINUX-NOT: "-lc"
// CHECK-NSAN-LINUX: libclang_rt.nsan.a"
// CHECK-NSAN-LINUX: "-lpthread" "-lrt" "-lm" "-ldl" "-lresolv"

// RUN: %clang -### %s 2>&1 --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=numerical -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-NSAN-SHARED-LINUX

// CHECK-NSAN-SHARED-LINUX: libclang_rt.nsan.so"
// CHECK-NSAN-SHARED-LINUX-NOT: "-lpthread"
// CHECK-NSAN-SHARED-LINUX-NOT: "-ldl"
// CHECK-NSAN-SHARED-LINUX-NOT: "--dynamic-list

// RUN: %clang -### %s 2>&1 --target=x86_64-unknown-linux -fsanitize=numerical,undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-NSAN-UBSAN

// CHECK-NSAN-UBSAN: "--whole-archive" "{{[^"]*}}libclang_rt.nsan.a" "--no-whole-archive"

// CFI by itself does not link runtime libraries.
// RUN: not %clang -fsanitize=cfi -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -rtlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-LINUX
// CHECK-CFI-LINUX: "{{.*}}ld{{(.exe)?}}"

// CFI with diagnostics links the UBSan runtime.
// RUN: not %clang -fsanitize=cfi -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     -### %s 2>&1\
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-DIAG-LINUX
// CHECK-CFI-DIAG-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-DIAG-LINUX: "--whole-archive" "{{[^"]*}}libclang_rt.ubsan_standalone.a" "--no-whole-archive"

// Cross-DSO CFI links the CFI runtime.
// RUN: not %clang -fsanitize=cfi -fsanitize-cfi-cross-dso -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-CROSS-DSO-LINUX
// CHECK-CFI-CROSS-DSO-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-LINUX: "--whole-archive" "{{[^"]*}}libclang_rt.cfi.a" "--no-whole-archive"
// CHECK-CFI-CROSS-DSO-LINUX: -export-dynamic

// Cross-DSO CFI with diagnostics links just the CFI runtime.
// RUN: not %clang -fsanitize=cfi -fsanitize-cfi-cross-dso -### %s 2>&1 \
// RUN:     -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-CROSS-DSO-DIAG-LINUX
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: "--whole-archive" "{{[^"]*}}libclang_rt.cfi_diag.a" "--no-whole-archive"
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: -export-dynamic

// Cross-DSO CFI on Android does not link runtime libraries.
// RUN: not %clang -fsanitize=cfi -fsanitize-cfi-cross-dso -### %s 2>&1 \
// RUN:     --target=aarch64-linux-android -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-CROSS-DSO-ANDROID
// CHECK-CFI-CROSS-DSO-ANDROID: "{{.*}}ld{{(.exe)?}}"

// Cross-DSO CFI with diagnostics on Android links just the UBSAN runtime.
// RUN: not %clang -fsanitize=cfi -fsanitize-cfi-cross-dso -### %s 2>&1 \
// RUN:     -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     --target=aarch64-linux-android -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-CROSS-DSO-DIAG-ANDROID
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "{{[^"]*}}libclang_rt.ubsan_standalone.so"
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "--export-dynamic-symbol=__cfi_check"

// RUN: %clangxx -fsanitize=address -### %s 2>&1 \
// RUN:     -mmacos-version-min=10.6 \
// RUN:     --target=x86_64-apple-darwin13.4.0 -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-DARWIN106-CXX
// CHECK-ASAN-DARWIN106-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-DARWIN106-CXX: libclang_rt.asan_osx_dynamic.dylib
// CHECK-ASAN-DARWIN106-CXX-NOT: -lc++abi

// RUN: %clangxx -fsanitize=leak -### %s 2>&1 \
// RUN:     -mmacos-version-min=10.6 \
// RUN:     --target=x86_64-apple-darwin13.4.0 -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-LSAN-DARWIN106-CXX
// CHECK-LSAN-DARWIN106-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LSAN-DARWIN106-CXX: libclang_rt.lsan_osx_dynamic.dylib
// CHECK-LSAN-DARWIN106-CXX-NOT: -lc++abi

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SAFESTACK-LINUX
//
// CHECK-SAFESTACK-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SAFESTACK-LINUX-NOT: "-lc"
// CHECK-SAFESTACK-LINUX-NOT: whole-archive
// CHECK-SAFESTACK-LINUX: "-u" "__safestack_init"
// CHECK-SAFESTACK-LINUX: libclang_rt.safestack.a"
// CHECK-SAFESTACK-LINUX: "-lpthread"
// CHECK-SAFESTACK-LINUX: "-ldl"
// CHECK-SAFESTACK-LINUX: "-lresolv"

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-X86-64
// CHECK-SHADOWCALLSTACK-LINUX-X86-64-NOT: error:
// CHECK-SHADOWCALLSTACK-LINUX-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: not %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-AARCH64
// CHECK-SHADOWCALLSTACK-LINUX-AARCH64: '-fsanitize=shadow-call-stack' only allowed with '-ffixed-x18'

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=riscv32-unknown-elf -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-ELF-RISCV32
// CHECK-SHADOWCALLSTACK-ELF-RISCV32-NOT: error:
// CHECK-SHADOWCALLSTACK-ELF-RISCV32: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=riscv64-unknown-linux -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-RISCV64
// CHECK-SHADOWCALLSTACK-LINUX-RISCV64-NOT: error:
// CHECK-SHADOWCALLSTACK-LINUX-RISCV64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: %clang -target riscv64-linux-android -fsanitize=shadow-call-stack %s -### 2>&1 \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-ANDROID-RISCV64
// CHECK-SHADOWCALLSTACK-ANDROID-RISCV64-NOT: error:
// CHECK-SHADOWCALLSTACK-ANDROID-RISCV64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=riscv64-unknown-fuchsia -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-FUCHSIA-RISCV64
// CHECK-SHADOWCALLSTACK-FUCHSIA-RISCV64-NOT: error:
// CHECK-SHADOWCALLSTACK-FUCHSIA-RISCV64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux -fuse-ld=ld -ffixed-x18 \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18
// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=arm64-unknown-ios -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18
// CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18-NOT: error:
// CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux-android -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18-ANDROID
// CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18-ANDROID-NOT: error:
// CHECK-SHADOWCALLSTACK-LINUX-AARCH64-X18-ANDROID: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: not %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     --target=x86-unknown-linux -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-LINUX-X86
// CHECK-SHADOWCALLSTACK-LINUX-X86: error: unsupported option '-fsanitize=shadow-call-stack' for target 'x86-unknown-linux'

// RUN: %clang -fsanitize=shadow-call-stack -### %s 2>&1 \
// RUN:     -fsanitize=safe-stack --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHADOWCALLSTACK-SAFESTACK
// CHECK-SHADOWCALLSTACK-SAFESTACK-NOT: error:
// CHECK-SHADOWCALLSTACK-SAFESTACK: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SHADOWCALLSTACK-SAFESTACK: libclang_rt.safestack.a

// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-LINUX
// CHECK-CFI-STATS-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-STATS-LINUX: "--whole-archive" "{{[^"]*}}libclang_rt.stats_client.a" "--no-whole-archive"
// CHECK-CFI-STATS-LINUX-NOT: "--whole-archive"
// CHECK-CFI-STATS-LINUX: "{{[^"]*}}libclang_rt.stats.a"

// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=x86_64-apple-darwin -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-DARWIN
// CHECK-CFI-STATS-DARWIN: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-STATS-DARWIN: "{{[^"]*}}libclang_rt.stats_client_osx.a"
// CHECK-CFI-STATS-DARWIN: "{{[^"]*}}libclang_rt.stats_osx_dynamic.dylib"

// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=x86_64-pc-windows \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-WIN64
// CHECK-CFI-STATS-WIN64: "--dependent-lib=clang_rt.stats_client{{(-x86_64)?}}.lib"
// CHECK-CFI-STATS-WIN64: "--dependent-lib=clang_rt.stats{{(-x86_64)?}}.lib"
// CHECK-CFI-STATS-WIN64: "--linker-option=/include:__sanitizer_stats_register"

// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=i686-pc-windows \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-WIN32
// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=i686-pc-windows \
// RUN:     -fno-rtlib-defaultlib \
// RUN:     -frtlib-defaultlib \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-WIN32
// CHECK-CFI-STATS-WIN32: "--dependent-lib=clang_rt.stats_client{{(-i386)?}}.lib"
// CHECK-CFI-STATS-WIN32: "--dependent-lib=clang_rt.stats{{(-i386)?}}.lib"
// CHECK-CFI-STATS-WIN32: "--linker-option=/include:___sanitizer_stats_register"

// RUN: not %clang -fsanitize=cfi -fsanitize-stats -### %s 2>&1 \
// RUN:     --target=i686-pc-windows \
// RUN:     -fno-rtlib-defaultlib \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-CFI-STATS-WIN32-NODEF
// CHECK-CFI-STATS-WIN32-NODEF-NOT: "--dependent-lib=clang_rt.stats_client{{(-i386)?}}.lib"
// CHECK-CFI-STATS-WIN32-NODEF-NOT: "--dependent-lib=clang_rt.stats{{(-i386)?}}.lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SAFESTACK-ANDROID-ARM
//
// CHECK-SAFESTACK-ANDROID-ARM: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: %clang -### %s -shared 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SAFESTACK-SHARED-ANDROID-ARM
//
// CHECK-SAFESTACK-SHARED-ANDROID-ARM: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-linux-android -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SAFESTACK-ANDROID-AARCH64
//
// CHECK-SAFESTACK-ANDROID-AARCH64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"

// RUN: not %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-PS4
// CHECK-UBSAN-PS4: --dependent-lib=libSceDbgUBSanitizer_stub_weak.a
// CHECK-UBSAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-UBSAN-PS4: -lSceDbgUBSanitizer_stub_weak

// RUN: not %clang -fsanitize=undefined -### %s 2>&1 \
// RUN:     --target=x86_64-sie-ps5 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-UBSAN-PS5
// CHECK-UBSAN-PS5: --dependent-lib=libSceUBSanitizer_nosubmission_stub_weak.a
// CHECK-UBSAN-PS5: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-UBSAN-PS5: -lSceUBSanitizer_nosubmission_stub_weak

// RUN: not %clang -fsanitize=address -### %s 2>&1 \
// RUN:     --target=x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-PS4
// CHECK-ASAN-PS4: --dependent-lib=libSceDbgAddressSanitizer_stub_weak.a
// CHECK-ASAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-ASAN-PS4: -lSceDbgAddressSanitizer_stub_weak

// RUN: not %clang -fsanitize=address -### %s 2>&1 \
// RUN:     --target=x86_64-sie-ps5 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-ASAN-PS5
// CHECK-ASAN-PS5: --dependent-lib=libSceAddressSanitizer_nosubmission_stub_weak.a
// CHECK-ASAN-PS5: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-ASAN-PS5: -lSceAddressSanitizer_nosubmission_stub_weak

// RUN: not %clang -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-AUBSAN-PS4
// CHECK-AUBSAN-PS4-NOT: --dependent-lib=libSceDbgUBSanitizer_stub_weak.a
// CHECK-AUBSAN-PS4: --dependent-lib=libSceDbgAddressSanitizer_stub_weak.a
// CHECK-AUBSAN-PS4-NOT: --dependent-lib=libSceDbgUBSanitizer_stub_weak.a
// CHECK-AUBSAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-AUBSAN-PS4: -lSceDbgAddressSanitizer_stub_weak

// RUN: not %clang -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-sie-ps5 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | %{filecheck} --check-prefix=CHECK-AUBSAN-PS5
// CHECK-AUBSAN-PS5-NOT: --dependent-lib=libSceUBSanitizer_nosubmission_stub_weak.a
// CHECK-AUBSAN-PS5: --dependent-lib=libSceAddressSanitizer_nosubmission_stub_weak.a
// CHECK-AUBSAN-PS5-NOT: --dependent-lib=libSceUBSanitizer_nosubmission_stub_weak.a
// CHECK-AUBSAN-PS5: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-AUBSAN-PS5: -lSceAddressSanitizer_nosubmission_stub_weak

// RUN: not %clang -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:     -nostdlib \
// RUN:   | %{filecheck} --check-prefix=CHECK-NOLIB-PS4
// CHECK-NOLIB-PS4-NOT: SceDbgAddressSanitizer_stub_weak

// RUN: not %clang -fsanitize=address,undefined -### %s 2>&1 \
// RUN:     --target=x86_64-sie-ps5 -fuse-ld=ld \
// RUN:     -shared \
// RUN:     -nostdlib \
// RUN:   | %{filecheck} --check-prefix=CHECK-NOLIB-PS5
// CHECK-NOLIB-PS5-NOT: SceAddressSanitizer_nosubmission_stub_weak

// RUN: %clang -fsanitize=scudo -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SCUDO-LINUX
// CHECK-SCUDO-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-SCUDO-LINUX: "--whole-archive" "{{.*}}libclang_rt.scudo_standalone.a" "--no-whole-archive"
// CHECK-SCUDO-LINUX-NOT: "-lstdc++"
// CHECK-SCUDO-LINUX: "-lpthread"
// CHECK-SCUDO-LINUX: "-ldl"
// CHECK-SCUDO-LINUX: "-lresolv"

// RUN: %clang -### %s -o %t.so -shared 2>&1 \
// RUN:     --target=i386-unknown-linux -fuse-ld=ld -fsanitize=scudo -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SCUDO-SHARED-LINUX
//
// CHECK-SCUDO-SHARED-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SCUDO-SHARED-LINUX-NOT: "-lc"
// CHECK-SCUDO-SHARED-LINUX: libclang_rt.scudo_standalone.so"
// CHECK-SCUDO-SHARED-LINUX-NOT: "-lpthread"
// CHECK-SCUDO-SHARED-LINUX-NOT: "-lrt"
// CHECK-SCUDO-SHARED-LINUX-NOT: "-ldl"
// CHECK-SCUDO-SHARED-LINUX-NOT: "-lresolv"
// CHECK-SCUDO-SHARED-LINUX-NOT: "--export-dynamic"
// CHECK-SCUDO-SHARED-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=scudo \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | %{filecheck} --check-prefix=CHECK-SCUDO-ANDROID
//
// CHECK-SCUDO-ANDROID: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-SCUDO-ANDROID-NOT: "-lc"
// CHECK-SCUDO-ANDROID: "-pie"
// CHECK-SCUDO-ANDROID-NOT: "-lpthread"
// CHECK-SCUDO-ANDROID-NOT: "-lresolv"
// CHECK-SCUDO-ANDROID: libclang_rt.scudo_standalone.so"
// CHECK-SCUDO-ANDROID-NOT: "-lpthread"
// CHECK-SCUDO-ANDROID-NOT: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -fuse-ld=ld -fsanitize=scudo \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static-libsan \
// RUN:   | %{filecheck} --check-prefix=CHECK-SCUDO-ANDROID-STATIC
// CHECK-SCUDO-ANDROID-STATIC: "{{(.*[^.0-9A-Z_a-z])?}}ld.lld{{(.exe)?}}"
// CHECK-SCUDO-ANDROID-STATIC: "-pie"
// CHECK-SCUDO-ANDROID-STATIC: "--whole-archive" "{{.*}}libclang_rt.scudo_standalone.a" "--no-whole-archive"
// CHECK-SCUDO-ANDROID-STATIC-NOT: "-lstdc++"
// CHECK-SCUDO-ANDROID-STATIC-NOT: "-lpthread"
// CHECK-SCUDO-ANDROID-STATIC-NOT: "-lrt"
// CHECK-SCUDO-ANDROID-STATIC-NOT: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-HWASAN-X86-64-LINUX
//
// CHECK-HWASAN-X86-64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-HWASAN-X86-64-LINUX-NOT: "-lc"
// CHECK-HWASAN-X86-64-LINUX: libclang_rt.hwasan.a"
// CHECK-HWASAN-X86-64-LINUX-NOT: "--export-dynamic"
// CHECK-HWASAN-X86-64-LINUX: "--dynamic-list={{.*}}libclang_rt.hwasan.a.syms"
// CHECK-HWASAN-X86-64-LINUX-NOT: "--export-dynamic"
// CHECK-HWASAN-X86-64-LINUX: "-lpthread"
// CHECK-HWASAN-X86-64-LINUX: "-lrt"
// CHECK-HWASAN-X86-64-LINUX: "-ldl"
// CHECK-HWASAN-X86-64-LINUX: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -shared-libsan -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHARED-HWASAN-X86-64-LINUX
//
// CHECK-SHARED-HWASAN-X86-64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "-lc"
// CHECK-SHARED-HWASAN-X86-64-LINUX: libclang_rt.hwasan.so"
// CHECK-SHARED-HWASAN-X86-64-LINUX: libclang_rt.hwasan-preinit.a"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "-lpthread"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "-lrt"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "-ldl"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "-lresolv"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "--export-dynamic"
// CHECK-SHARED-HWASAN-X86-64-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s -o %t.so -shared 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -shared-libsan -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-DSO-SHARED-HWASAN-X86-64-LINUX
//
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "-lc"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX: libclang_rt.hwasan.so"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "-lpthread"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "-lrt"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "-ldl"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "-lresolv"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "--export-dynamic"
// CHECK-DSO-SHARED-HWASAN-X86-64-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-HWASAN-AARCH64-LINUX
//
// CHECK-HWASAN-AARCH64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-HWASAN-AARCH64-LINUX-NOT: "-lc"
// CHECK-HWASAN-AARCH64-LINUX: libclang_rt.hwasan.a"
// CHECK-HWASAN-AARCH64-LINUX-NOT: "--export-dynamic"
// CHECK-HWASAN-AARCH64-LINUX: "--dynamic-list={{.*}}libclang_rt.hwasan.a.syms"
// CHECK-HWASAN-AARCH64-LINUX-NOT: "--export-dynamic"
// CHECK-HWASAN-AARCH64-LINUX: "-lpthread"
// CHECK-HWASAN-AARCH64-LINUX: "-lrt"
// CHECK-HWASAN-AARCH64-LINUX: "-ldl"
// CHECK-HWASAN-AARCH64-LINUX: "-lresolv"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-SHARED-HWASAN-AARCH64-LINUX
//
// CHECK-SHARED-HWASAN-AARCH64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lc"
// CHECK-SHARED-HWASAN-AARCH64-LINUX: libclang_rt.hwasan.so"
// CHECK-SHARED-HWASAN-AARCH64-LINUX: libclang_rt.hwasan-preinit.a"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lpthread"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lrt"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "-ldl"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lresolv"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "--export-dynamic"
// CHECK-SHARED-HWASAN-AARCH64-LINUX-NOT: "--dynamic-list"

// RUN: %clang -### %s -o %t.so -shared 2>&1 \
// RUN:     --target=aarch64-unknown-linux -fuse-ld=ld -fsanitize=hwaddress \
// RUN:     -shared-libsan -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | %{filecheck} --check-prefix=CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX
//
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lc"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX: libclang_rt.hwasan.so"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lpthread"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lrt"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "-ldl"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "-lresolv"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "--export-dynamic"
// CHECK-DSO-SHARED-HWASAN-AARCH64-LINUX-NOT: "--dynamic-list"
