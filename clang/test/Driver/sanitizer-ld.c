// Test sanitizers ld flags.

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-LINUX %s
//
// CHECK-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-NOT: "-lc"
// CHECK-ASAN-LINUX: libclang_rt.asan-i386.a"
// CHECK-ASAN-LINUX-NOT: "-export-dynamic"
// CHECK-ASAN-LINUX: "--dynamic-list={{.*}}libclang_rt.asan-i386.a.syms"
// CHECK-ASAN-LINUX-NOT: "-export-dynamic"
// CHECK-ASAN-LINUX: "-lpthread"
// CHECK-ASAN-LINUX: "-lrt"
// CHECK-ASAN-LINUX: "-ldl"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED-ASAN-LINUX %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED-ASAN-LINUX %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -fsanitize=address \
// RUN:     -shared-libsan -static-libsan -shared-libasan             \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED-ASAN-LINUX %s
//
// CHECK-SHARED-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lc"
// CHECK-SHARED-ASAN-LINUX-NOT: libclang_rt.asan-i386.a"
// CHECK-SHARED-ASAN-LINUX: libclang_rt.asan-i386.so"
// CHECK-SHARED-ASAN-LINUX: "-whole-archive" "{{.*}}libclang_rt.asan-preinit-i386.a" "-no-whole-archive"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lpthread"
// CHECK-SHARED-ASAN-LINUX-NOT: "-lrt"
// CHECK-SHARED-ASAN-LINUX-NOT: "-ldl"
// CHECK-SHARED-ASAN-LINUX-NOT: "-export-dynamic"
// CHECK-SHARED-ASAN-LINUX-NOT: "--dynamic-list"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.so -shared 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -fsanitize=address -shared-libsan \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DSO-SHARED-ASAN-LINUX %s
//
// CHECK-DSO-SHARED-ASAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lc"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: libclang_rt.asan-i386.a"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "libclang_rt.asan-preinit-i386.a"
// CHECK-DSO-SHARED-ASAN-LINUX: libclang_rt.asan-i386.so"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lpthread"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-lrt"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-ldl"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "-export-dynamic"
// CHECK-DSO-SHARED-ASAN-LINUX-NOT: "--dynamic-list"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-freebsd -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-FREEBSD %s
//
// CHECK-ASAN-FREEBSD: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-FREEBSD-NOT: "-lc"
// CHECK-ASAN-FREEBSD-NOT: libclang_rt.asan_cxx
// CHECK-ASAN-FREEBSD: freebsd{{/|\\+}}libclang_rt.asan-i386.a"
// CHECK-ASAN-FREEBSD-NOT: libclang_rt.asan_cxx
// CHECK-ASAN-FREEBSD-NOT: "--dynamic-list"
// CHECK-ASAN-FREEBSD: "-export-dynamic"
// CHECK-ASAN-FREEBSD: "-lpthread"
// CHECK-ASAN-FREEBSD: "-lrt"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-freebsd -fuse-ld=ld -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-FREEBSD-LDL %s
//
// CHECK-ASAN-FREEBSD-LDL: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-FREEBSD-LDL-NOT: "-ldl"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -stdlib=platform -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/empty_resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-LINUX-CXX %s
//
// CHECK-ASAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-CXX-NOT: "-lc"
// CHECK-ASAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.asan-i386.a" "-no-whole-archive"
// CHECK-ASAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.asan_cxx-i386.a" "-no-whole-archive"
// CHECK-ASAN-LINUX-CXX-NOT: "--dynamic-list"
// CHECK-ASAN-LINUX-CXX: "-export-dynamic"
// CHECK-ASAN-LINUX-CXX: stdc++
// CHECK-ASAN-LINUX-CXX: "-lpthread"
// CHECK-ASAN-LINUX-CXX: "-lrt"
// CHECK-ASAN-LINUX-CXX: "-ldl"

// RUN: %clang -no-canonical-prefixes %s -### -o /dev/null -fsanitize=address \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree -lstdc++ -static 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-LINUX-CXX-STATIC %s
//
// CHECK-ASAN-LINUX-CXX-STATIC: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-LINUX-CXX-STATIC-NOT: stdc++
// CHECK-ASAN-LINUX-CXX-STATIC: "-whole-archive" "{{.*}}libclang_rt.asan-i386.a" "-no-whole-archive"
// CHECK-ASAN-LINUX-CXX-STATIC: stdc++

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-gnueabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ARM %s
//
// CHECK-ASAN-ARM: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ARM-NOT: "-lc"
// CHECK-ASAN-ARM: libclang_rt.asan-arm.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv7l-linux-gnueabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ARMv7 %s
//
// CHECK-ASAN-ARMv7: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ARMv7-NOT: "-lc"
// CHECK-ASAN-ARMv7: libclang_rt.asan-arm.a"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ANDROID %s
//
// CHECK-ASAN-ANDROID: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-NOT: "-lc"
// CHECK-ASAN-ANDROID-NOT: "-pie"
// CHECK-ASAN-ANDROID-NOT: "-lpthread"
// CHECK-ASAN-ANDROID: libclang_rt.asan-arm-android.so"
// CHECK-ASAN-ANDROID-NOT: "-lpthread"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ANDROID-STATICLIBASAN %s
//
// CHECK-ASAN-ANDROID-STATICLIBASAN: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-STATICLIBASAN: libclang_rt.asan-arm-android.a"
// CHECK-ASAN-ANDROID-STATICLIBASAN: "-lpthread"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=undefined \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-ANDROID %s
//
// CHECK-UBSAN-ANDROID: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-UBSAN-ANDROID-NOT: "-lc"
// CHECK-UBSAN-ANDROID-NOT: "-pie"
// CHECK-UBSAN-ANDROID-NOT: "-lpthread"
// CHECK-UBSAN-ANDROID: libclang_rt.ubsan_standalone-arm-android.so"
// CHECK-UBSAN-ANDROID-NOT: "-lpthread"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=undefined \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-ANDROID-STATICLIBASAN %s
//
// CHECK-UBSAN-ANDROID-STATICLIBASAN: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-UBSAN-ANDROID-STATICLIBASAN: libclang_rt.ubsan_standalone-arm-android.a"
// CHECK-UBSAN-ANDROID-STATICLIBASAN: "-lpthread"

//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i686-linux-android -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ANDROID-X86 %s
//
// CHECK-ASAN-ANDROID-X86: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-X86-NOT: "-lc"
// CHECK-ASAN-ANDROID-X86-NOT: "-pie"
// CHECK-ASAN-ANDROID-X86-NOT: "-lpthread"
// CHECK-ASAN-ANDROID-X86: libclang_rt.asan-i686-android.so"
// CHECK-ASAN-ANDROID-X86-NOT: "-lpthread"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ANDROID-SHARED-LIBASAN %s
//
// CHECK-ASAN-ANDROID-SHARED-LIBASAN-NOT: argument unused during compilation: '-shared-libsan'
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-ANDROID-SHARED %s
//
// CHECK-ASAN-ANDROID-SHARED: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-ANDROID-SHARED-NOT: "-lc"
// CHECK-ASAN-ANDROID-SHARED: libclang_rt.asan-arm-android.so"
// CHECK-ASAN-ANDROID-SHARED-NOT: "-lpthread"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target sparcel-myriad-rtems-elf -fuse-ld=ld -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_myriad_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-MYRIAD %s
//
// CHECK-ASAN-MYRIAD: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ASAN-MYRIAD-NOT: "-lc"
// CHECK-ASAN-MYRIAD: libclang_rt.asan-sparcel.a"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld -stdlib=platform -lstdc++ \
// RUN:     -fsanitize=thread \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-TSAN-LINUX-CXX %s
//
// CHECK-TSAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-TSAN-LINUX-CXX-NOT: stdc++
// CHECK-TSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.tsan-x86_64.a" "-no-whole-archive"
// CHECK-TSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan-x86_64.a.syms"
// CHECK-TSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.tsan_cxx-x86_64.a" "-no-whole-archive"
// CHECK-TSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.tsan_cxx-x86_64.a.syms"
// CHECK-TSAN-LINUX-CXX-NOT: "-export-dynamic"
// CHECK-TSAN-LINUX-CXX: stdc++
// CHECK-TSAN-LINUX-CXX: "-lpthread"
// CHECK-TSAN-LINUX-CXX: "-lrt"
// CHECK-TSAN-LINUX-CXX: "-ldl"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld -stdlib=platform -lstdc++ \
// RUN:     -fsanitize=memory \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MSAN-LINUX-CXX %s
//
// CHECK-MSAN-LINUX-CXX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-MSAN-LINUX-CXX-NOT: stdc++
// CHECK-MSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.msan-x86_64.a" "-no-whole-archive"
// CHECK-MSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan-x86_64.a.syms"
// CHECK-MSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.msan_cxx-x86_64.a" "-no-whole-archive"
// CHECK-MSAN-LINUX-CXX: "--dynamic-list={{.*}}libclang_rt.msan_cxx-x86_64.a.syms"
// CHECK-MSAN-LINUX-CXX-NOT: "-export-dynamic"
// CHECK-MSAN-LINUX-CXX: stdc++
// CHECK-MSAN-LINUX-CXX: "-lpthread"
// CHECK-MSAN-LINUX-CXX: "-lrt"
// CHECK-MSAN-LINUX-CXX: "-ldl"

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX %s

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -static-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX %s

// CHECK-UBSAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-NOT: libclang_rt.ubsan_standalone_cxx
// CHECK-UBSAN-LINUX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone-i386.a" "-no-whole-archive"
// CHECK-UBSAN-LINUX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-NOT: libclang_rt.ubsan_standalone_cxx
// CHECK-UBSAN-LINUX-NOT: "-lstdc++"
// CHECK-UBSAN-LINUX: "-lpthread"

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN %s

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -static-libsan -shared-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN %s

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared -shared-libsan \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-SHAREDLIBASAN %s

// CHECK-UBSAN-LINUX-SHAREDLIBASAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-SHAREDLIBASAN: "{{.*}}libclang_rt.ubsan_standalone-i386.so{{.*}}"

// RUN: %clang -fsanitize=undefined -fsanitize-link-c++-runtime %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-LINK-CXX %s
// CHECK-UBSAN-LINUX-LINK-CXX-NOT: "-lstdc++"
// CHECK-UBSAN-LINUX-LINK-CXX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx-i386.a" "-no-whole-archive"
// CHECK-UBSAN-LINUX-LINK-CXX-NOT: "-lstdc++"

// RUN: %clangxx -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-CXX %s
// CHECK-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-CXX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone-i386.a" "-no-whole-archive"
// CHECK-UBSAN-LINUX-CXX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone_cxx-i386.a" "-no-whole-archive"
// CHECK-UBSAN-LINUX-CXX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-CXX: "-lstdc++"
// CHECK-UBSAN-LINUX-CXX-NOT: libclang_rt.asan
// CHECK-UBSAN-LINUX-CXX: "-lpthread"

// RUN: %clang -fsanitize=undefined -fsanitize-minimal-runtime %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-MINIMAL-LINUX %s
// CHECK-UBSAN-MINIMAL-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-MINIMAL-LINUX: "-whole-archive" "{{.*}}libclang_rt.ubsan_minimal-i386.a" "-no-whole-archive"
// CHECK-UBSAN-MINIMAL-LINUX: "-lpthread"

// RUN: %clang -fsanitize=undefined -fsanitize-minimal-runtime %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-apple-darwin -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-MINIMAL-DARWIN %s
// CHECK-UBSAN-MINIMAL-DARWIN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-MINIMAL-DARWIN: "{{.*}}libclang_rt.ubsan_minimal_osx_dynamic.dylib"

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-apple-darwin -fuse-ld=ld -static-libsan \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-STATIC-DARWIN %s
// CHECK-UBSAN-STATIC-DARWIN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-STATIC-DARWIN: "{{.*}}libclang_rt.ubsan_osx.a"

// RUN: %clang -fsanitize=address,undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-UBSAN-LINUX %s
// CHECK-ASAN-UBSAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-UBSAN-LINUX: "-whole-archive" "{{.*}}libclang_rt.asan-i386.a" "-no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-NOT: libclang_rt.ubsan
// CHECK-ASAN-UBSAN-LINUX-NOT: "-lstdc++"
// CHECK-ASAN-UBSAN-LINUX: "-lpthread"

// RUN: %clangxx -fsanitize=address,undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld -stdlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-UBSAN-LINUX-CXX %s
// CHECK-ASAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.asan-i386.a" "-no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.asan_cxx-i386.a" "-no-whole-archive"
// CHECK-ASAN-UBSAN-LINUX-CXX-NOT: libclang_rt.ubsan
// CHECK-ASAN-UBSAN-LINUX-CXX: "-lstdc++"
// CHECK-ASAN-UBSAN-LINUX-CXX: "-lpthread"

// RUN: %clangxx -fsanitize=memory,undefined %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MSAN-UBSAN-LINUX-CXX %s
// CHECK-MSAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-MSAN-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.msan-x86_64.a" "-no-whole-archive"
// CHECK-MSAN-UBSAN-LINUX-CXX-NOT: libclang_rt.ubsan

// RUN: %clangxx -fsanitize=thread,undefined %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-TSAN-UBSAN-LINUX-CXX %s
// CHECK-TSAN-UBSAN-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-TSAN-UBSAN-LINUX-CXX: "-whole-archive" "{{.*}}libclang_rt.tsan-x86_64.a" "-no-whole-archive"
// CHECK-TSAN-UBSAN-LINUX-CXX-NOT: libclang_rt.ubsan

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-LINUX-SHARED %s
// CHECK-UBSAN-LINUX-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-LINUX-SHARED-NOT: --export-dynamic
// CHECK-UBSAN-LINUX-SHARED-NOT: --dynamic-list
// CHECK-UBSAN-LINUX-SHARED-NOT: libclang_rt.ubsan

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld -fsanitize=leak \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LSAN-LINUX %s
//
// CHECK-LSAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LSAN-LINUX-NOT: "-lc"
// CHECK-LSAN-LINUX-NOT: libclang_rt.ubsan
// CHECK-LSAN-LINUX: libclang_rt.lsan-x86_64.a"
// CHECK-LSAN-LINUX: "-lpthread"
// CHECK-LSAN-LINUX: "-ldl"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:  -target x86_64-unknown-linux -fuse-ld=ld -fsanitize=leak -fsanitize-coverage=func \
// RUN:  --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LSAN-COV-LINUX %s
//
// CHECK-LSAN-COV-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LSAN-COV-LINUX-NOT: "-lc"
// CHECK-LSAN-COV-LINUX-NOT: libclang_rt.ubsan
// CHECK-LSAV-COV-LINUX: libclang_rt.lsan-x86_64.a"
// CHECK-LSAN-COV-LINUX-NOT: libclang_rt.ubsan
// CHECK-LSAN-COV-LINUX: "-lpthread"
// CHECK-LSAN-COV-LINUX: "-ldl"

// RUN: %clang -fsanitize=leak,address %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LSAN-ASAN-LINUX %s
// CHECK-LSAN-ASAN-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LSAN-ASAN-LINUX-NOT: libclang_rt.lsan
// CHECK-LSAN-ASAN-LINUX: libclang_rt.asan-x86_64
// CHECK-LSAN-ASAN-LINUX-NOT: libclang_rt.lsan

// RUN: %clang -fsanitize=address -fsanitize-coverage=func %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-COV-LINUX %s
// CHECK-ASAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-COV-LINUX: "-whole-archive" "{{.*}}libclang_rt.asan-x86_64.a" "-no-whole-archive"
// CHECK-ASAN-COV-LINUX-NOT: libclang_rt.ubsan
// CHECK-ASAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-ASAN-COV-LINUX: "-lpthread"

// RUN: %clang -fsanitize=memory -fsanitize-coverage=func %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MSAN-COV-LINUX %s
// CHECK-MSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-MSAN-COV-LINUX: "-whole-archive" "{{.*}}libclang_rt.msan-x86_64.a" "-no-whole-archive"
// CHECK-MSAN-COV-LINUX-NOT: libclang_rt.ubsan
// CHECK-MSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-MSAN-COV-LINUX: "-lpthread"

// RUN: %clang -fsanitize=dataflow -fsanitize-coverage=func %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DFSAN-COV-LINUX %s
// CHECK-DFSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-DFSAN-COV-LINUX: "-whole-archive" "{{.*}}libclang_rt.dfsan-x86_64.a" "-no-whole-archive"
// CHECK-DFSAN-COV-LINUX-NOT: libclang_rt.ubsan
// CHECK-DFSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-DFSAN-COV-LINUX: "-lpthread"

// RUN: %clang -fsanitize=undefined -fsanitize-coverage=func %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-COV-LINUX %s
// CHECK-UBSAN-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-COV-LINUX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone-x86_64.a" "-no-whole-archive"
// CHECK-UBSAN-COV-LINUX-NOT: "-lstdc++"
// CHECK-UBSAN-COV-LINUX: "-lpthread"

// RUN: %clang -fsanitize-coverage=func %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-COV-LINUX %s
// CHECK-COV-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-COV-LINUX: "-whole-archive" "{{.*}}libclang_rt.ubsan_standalone-x86_64.a" "-no-whole-archive"
// CHECK-COV-LINUX-NOT: "-lstdc++"
// CHECK-COV-LINUX: "-lpthread"

// CFI by itself does not link runtime libraries.
// RUN: %clang -fsanitize=cfi %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-LINUX %s
// CHECK-CFI-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-LINUX-NOT: libclang_rt.

// CFI with diagnostics links the UBSan runtime.
// RUN: %clang -fsanitize=cfi -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     %s -### -o %t.o 2>&1\
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-DIAG-LINUX %s
// CHECK-CFI-DIAG-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-DIAG-LINUX: "-whole-archive" "{{[^"]*}}libclang_rt.ubsan_standalone-x86_64.a" "-no-whole-archive"

// Cross-DSO CFI links the CFI runtime.
// RUN: %clang -fsanitize=cfi -fsanitize-cfi-cross-dso %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-CROSS-DSO-LINUX %s
// CHECK-CFI-CROSS-DSO-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-LINUX: "-whole-archive" "{{[^"]*}}libclang_rt.cfi-x86_64.a" "-no-whole-archive"
// CHECK-CFI-CROSS-DSO-LINUX: -export-dynamic

// Cross-DSO CFI with diagnostics links just the CFI runtime.
// RUN: %clang -fsanitize=cfi -fsanitize-cfi-cross-dso %s -### -o %t.o 2>&1 \
// RUN:     -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-CROSS-DSO-DIAG-LINUX %s
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: "-whole-archive" "{{[^"]*}}libclang_rt.cfi_diag-x86_64.a" "-no-whole-archive"
// CHECK-CFI-CROSS-DSO-DIAG-LINUX: -export-dynamic

// Cross-DSO CFI on Android does not link runtime libraries.
// RUN: %clang -fsanitize=cfi -fsanitize-cfi-cross-dso %s -### -o %t.o 2>&1 \
// RUN:     -target aarch64-linux-android -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-CROSS-DSO-ANDROID %s
// CHECK-CFI-CROSS-DSO-ANDROID: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-ANDROID-NOT: libclang_rt.

// Cross-DSO CFI with diagnostics on Android links just the UBSAN runtime.
// RUN: %clang -fsanitize=cfi -fsanitize-cfi-cross-dso %s -### -o %t.o 2>&1 \
// RUN:     -fno-sanitize-trap=cfi -fsanitize-recover=cfi \
// RUN:     -target aarch64-linux-android -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-CROSS-DSO-DIAG-ANDROID %s
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "{{[^"]*}}libclang_rt.ubsan_standalone-aarch64-android.so"
// CHECK-CFI-CROSS-DSO-DIAG-ANDROID: "-export-dynamic-symbol=__cfi_check"

// RUN: %clangxx -fsanitize=address %s -### -o %t.o 2>&1 \
// RUN:     -mmacosx-version-min=10.6 \
// RUN:     -target x86_64-apple-darwin13.4.0 -fuse-ld=ld -stdlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-DARWIN106-CXX %s
// CHECK-ASAN-DARWIN106-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-DARWIN106-CXX: libclang_rt.asan_osx_dynamic.dylib
// CHECK-ASAN-DARWIN106-CXX-NOT: -lc++abi

// RUN: %clangxx -fsanitize=leak %s -### -o %t.o 2>&1 \
// RUN:     -mmacosx-version-min=10.6 \
// RUN:     -target x86_64-apple-darwin13.4.0 -fuse-ld=ld -stdlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LSAN-DARWIN106-CXX %s
// CHECK-LSAN-DARWIN106-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LSAN-DARWIN106-CXX: libclang_rt.lsan_osx_dynamic.dylib
// CHECK-LSAN-DARWIN106-CXX-NOT: -lc++abi

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SAFESTACK-LINUX %s
//
// CHECK-SAFESTACK-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SAFESTACK-LINUX-NOT: "-lc"
// CHECK-SAFESTACK-LINUX-NOT: whole-archive
// CHECK-SAFESTACK-LINUX: libclang_rt.safestack-x86_64.a"
// CHECK-SAFESTACK-LINUX: "-u" "__safestack_init"
// CHECK-SAFESTACK-LINUX: "-lpthread"
// CHECK-SAFESTACK-LINUX: "-ldl"

// RUN: %clang -fsanitize=cfi -fsanitize-stats %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-STATS-LINUX %s
// CHECK-CFI-STATS-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-STATS-LINUX: "-whole-archive" "{{[^"]*}}libclang_rt.stats_client-x86_64.a" "-no-whole-archive"
// CHECK-CFI-STATS-LINUX-NOT: "-whole-archive"
// CHECK-CFI-STATS-LINUX: "{{[^"]*}}libclang_rt.stats-x86_64.a"

// RUN: %clang -fsanitize=cfi -fsanitize-stats %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-apple-darwin -fuse-ld=ld \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-STATS-DARWIN %s
// CHECK-CFI-STATS-DARWIN: "{{.*}}ld{{(.exe)?}}"
// CHECK-CFI-STATS-DARWIN: "{{[^"]*}}libclang_rt.stats_client_osx.a"
// CHECK-CFI-STATS-DARWIN: "{{[^"]*}}libclang_rt.stats_osx_dynamic.dylib"

// RUN: %clang -fsanitize=cfi -fsanitize-stats %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-pc-windows \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-STATS-WIN64 %s
// CHECK-CFI-STATS-WIN64: "--dependent-lib={{[^"]*}}clang_rt.stats_client-x86_64.lib"
// CHECK-CFI-STATS-WIN64: "--dependent-lib={{[^"]*}}clang_rt.stats-x86_64.lib"
// CHECK-CFI-STATS-WIN64: "--linker-option=/include:__sanitizer_stats_register"

// RUN: %clang -fsanitize=cfi -fsanitize-stats %s -### -o %t.o 2>&1 \
// RUN:     -target i686-pc-windows \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CFI-STATS-WIN32 %s
// CHECK-CFI-STATS-WIN32: "--dependent-lib={{[^"]*}}clang_rt.stats_client-i386.lib"
// CHECK-CFI-STATS-WIN32: "--dependent-lib={{[^"]*}}clang_rt.stats-i386.lib"
// CHECK-CFI-STATS-WIN32: "--linker-option=/include:___sanitizer_stats_register"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SAFESTACK-ANDROID-ARM %s
//
// CHECK-SAFESTACK-ANDROID-ARM: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SAFESTACK-ANDROID-ARM-NOT: libclang_rt.safestack

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o -shared 2>&1 \
// RUN:     -target arm-linux-androideabi -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SAFESTACK-SHARED-ANDROID-ARM %s
//
// CHECK-SAFESTACK-SHARED-ANDROID-ARM: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SAFESTACK-SHARED-ANDROID-ARM-NOT: libclang_rt.safestack

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target aarch64-linux-android -fuse-ld=ld -fsanitize=safe-stack \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SAFESTACK-ANDROID-AARCH64 %s
//
// CHECK-SAFESTACK-ANDROID-AARCH64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-SAFESTACK-ANDROID-AARCH64-NOT: libclang_rt.safestack

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN-PS4 %s
// CHECK-UBSAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-UBSAN-PS4: -lSceDbgUBSanitizer_stub_weak

// RUN: %clang -fsanitize=address %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-PS4 %s
// CHECK-ASAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-ASAN-PS4: -lSceDbgAddressSanitizer_stub_weak

// RUN: %clang -fsanitize=address,undefined %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-scei-ps4 -fuse-ld=ld \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-AUBSAN-PS4 %s
// CHECK-AUBSAN-PS4: "{{.*}}ld{{(.gold)?(.exe)?}}"
// CHECK-AUBSAN-PS4: -lSceDbgAddressSanitizer_stub_weak

// RUN: %clang -fsanitize=efficiency-cache-frag %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:   | FileCheck --check-prefix=CHECK-ESAN-LINUX %s
// RUN: %clang -fsanitize=efficiency-working-set %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fuse-ld=ld \
// RUN:   | FileCheck --check-prefix=CHECK-ESAN-LINUX %s
//
// CHECK-ESAN-LINUX: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-ESAN-LINUX: libclang_rt.esan-x86_64.a
