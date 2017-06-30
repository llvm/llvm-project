// RUN: %clang -### --driver-mode=g++ -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -stdlib=libstdc++ -rtlib=platform -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-BASIC-LIBSTDCXX

// CHECK-BASIC-LIBSTDCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/gcc" "{{.*}}.o" "-lstdc++" "-lmingw32" "-lmingwex" "-lgcc" "-lmoldname" "-lmingw32" "-lmsvcrt" "-lgcc_s" "-lgcc"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -stdlib=libstdc++ -rtlib=compiler-rt -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-BASIC-LIBCXX

// CHECK-BASIC-LIBCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{[^"]*}}.o" "-lmsvcrt"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %s/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -rtlib=compiler-rt -stdlib=libstdc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-RTLIB

// CHECK-RTLIB: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-C-LIBCXX

// CHECK-C-LIBCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clangxx -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-LIBCXX

// CHECK-LIBCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}.o" "-lc++" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -shared -rtlib=compiler-rt -stdlib=libc++ -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SHARED

// CHECK-SHARED: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %s/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-NOSTARTFILES

// CHECK-NOSTARTFILES: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=ld -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -nodefaultlibs -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-STANDALONE

// CHECK-STANDALONE: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %/Inputs/Windows/ARM/8.1/usr/bin -stdlib=libstdc++ -shared -o shared.dll -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-LIBSTDCXX

// CHECK-LIBSTDCXX:  "-internal-isystem" "{{.*}}/usr/include/c++" "-internal-isystem" "{{.*}}/usr/include/c++/armv7--windows-itanium" "-internal-isystem" "{{.*}}/usr/include/c++/backwards"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-FUSE-LD

// CHECK-FUSE-LD: "{{.*}}lld-link2"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -fsanitize=address -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-ADDRESS

// CHECK-SANITIZE-ADDRESS: "-fsanitize=address"
// CHECK-SANITIZE-ADDRESS: "{{.*}}clang_rt.asan_dll_thunk-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -o test.exe -fsanitize=address -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-ADDRESS-EXE

// CHECK-SANITIZE-ADDRESS-EXE: "-fsanitize=address"
// CHECK-SANITIZE-ADDRESS-EXE: "{{.*}}clang_rt.asan_dynamic-arm.lib" "{{.*}}clang_rt.asan_dynamic_runtime_thunk-arm.lib" "--undefined" "__asan_seh_interceptor"

// RUN: %clang -### -target i686-windows-itanium -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -o test.exe -fsanitize=address -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-ADDRESS-EXE-X86

// CHECK-SANITIZE-ADDRESS-EXE-X86: "-fsanitize=address"
// CHECK-SANITIZE-ADDRESS-EXE-X86: "{{.*}}clang_rt.asan_dynamic-i686.lib" "{{.*}}clang_rt.asan_dynamic_runtime_thunk-i686.lib" "--undefined" "___asan_seh_interceptor"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -fsanitize=tsan -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-TSAN

// CHECK-SANITIZE-TSAN: error: unsupported argument 'tsan' to option 'fsanitize='
// CHECK-SANITIZE-TSAN-NOT: "-fsanitize={{.*}}"

// RUN: %clang -### -target armv7-windows-itanium -isystem-after "Windows Kits/10/Include/10.0.10586.0/ucrt" -isystem-after "Windows Kits/10/Include/10.0.10586.0/um" -isystem-after "Windows Kits/10/Include/10.0.10586.0/shared" -c %s -o /dev/null 2>&1 \
// RUN:     | FileCheck %s --check-prefix CHECK-ISYSTEM-AFTER
// CHECK-ISYSTEM-AFTER: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ISYSTEM-AFTER: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-ISYSTEM-AFTER: "-internal-isystem" "Windows Kits{{[/\\]}}10{{[/\\]}}Include{{[/\\]}}10.0.10586.0{{[/\\]}}ucrt"
// CHECK-ISYSTEM-AFTER: "-internal-isystem" "Windows Kits{{[/\\]}}10{{[/\\]}}Include{{[/\\]}}10.0.10586.0{{[/\\]}}um"
// CHECK-ISYSTEM-AFTER: "-internal-isystem" "Windows Kits{{[/\\]}}10{{[/\\]}}Include{{[/\\]}}10.0.10586.0{{[/\\]}}shared"

// RUN: %clang -### -target armv7-windows-itanium -nostdinc -isystem-after "Windows Kits/10/Include/10.0.10586.0/ucrt" -c %s -o /dev/null 2>&1 \
// RUN:     | FileCheck %s --check-prefix CHECK-NOSTDINC-ISYSTEM-AFTER
// CHECK-NOSTDINC-ISYSTEM-AFTER: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDINC-ISYSTEM-AFTER-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOSTDINC-ISYSTEM-AFTER: "-internal-isystem" "Windows Kits{{[/\\]}}10{{[/\\]}}Include{{[/\\]}}10.0.10586.0{{[/\\]}}ucrt"
