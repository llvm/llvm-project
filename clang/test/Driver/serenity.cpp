// UNSUPPORTED: system-windows

/// Check default header and linker paths for each supported triple.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   2>&1 | FileCheck %s --check-prefix=PATHS

// RUN: %clang -### %s --target=aarch64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   2>&1 | FileCheck %s --check-prefix=PATHS

// RUN: %clang -### %s --target=riscv64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   2>&1 | FileCheck %s --check-prefix=PATHS

// PATHS:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// PATHS-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// PATHS-SAME: "-internal-isystem" "[[RESOURCE]]/include"
// PATHS-SAME: "-internal-isystem" "[[SYSROOT]]/usr/include"
// PATHS:      "-L[[SYSROOT]]/usr/lib"

/// Check include paths with -nostdinc.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   -fsyntax-only -nostdinc 2>&1 | FileCheck %s --check-prefix=PATH_NOSTDINC
// PATH_NOSTDINC: "-nostdsysteminc" "-nobuiltininc"
// PATH_NOSTDINC-NOT: /include

/// Check include paths with -nobuiltininc.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   -fsyntax-only -nobuiltininc 2>&1 | FileCheck %s --check-prefix=PATH_NOBUILTIN
// PATH_NOBUILTIN: "-nobuiltininc"
// PATH_NOBUILTIN-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// PATH_NOBUILTIN-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// PATH_NOBUILTIN-NOT: "-internal-isystem" "[[RESOURCE]]/include"
// PATH_NOBUILTIN-SAME: "-internal-isystem" "[[SYSROOT]]/usr/include"

/// Check include paths with -nostdlibinc.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   -fsyntax-only -nostdlibinc 2>&1 | FileCheck %s --check-prefix=PATH_NOSTDLIBINC
// PATH_NOSTDLIBINC: "-nostdsysteminc"
// PATH_NOSTDLIBINC-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// PATH_NOSTDLIBINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// PATH_NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// PATH_NOSTDLIBINC-SAME: "-internal-isystem" "[[RESOURCE]]/include"
// PATH_NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include"

/// Check that PIC and PIE are enabled by default.
// RUN: %clang -c %s --target=x86_64-unkown-serenity -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s --target=aarch64-unkown-serenity -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s --target=riscv64-unkown-serenity -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2

// CHECK-PIE2: "-mrelocation-model" "pic"
// CHECK-PIE2-SAME: "-pic-level" "2"
// CHECK-PIE2-SAME: "-pic-is-pie"

/// Check default linker args for each supported triple.
// RUN: %clang -### %s --target=x86_64-unknown-serenity \
// RUN:   --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   2>&1 | FileCheck %s --check-prefix=SERENITY_X86_64,DEFAULT_LINKER
// SERENITY_X86_64: "-cc1" "-triple" "[[TRIPLE:x86_64-unknown-serenity]]"

// RUN: %clang -### %s --target=aarch64-unknown-serenity \
// RUN:   --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   2>&1 | FileCheck %s --check-prefix=SERENITY_AARCH64,DEFAULT_LINKER
// SERENITY_AARCH64: "-cc1" "-triple" "[[TRIPLE:aarch64-unknown-serenity]]"

// RUN: %clang -### %s --target=riscv64-unknown-serenity \
// RUN:   --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   2>&1 | FileCheck %s --check-prefix=SERENITY_RISCV64,DEFAULT_LINKER
// SERENITY_RISCV64: "-cc1" "-triple" "[[TRIPLE:riscv64-unknown-serenity]]"

// DEFAULT_LINKER: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEFAULT_LINKER: ld.lld"
// DEFAULT_LINKER-SAME: "-pie"
// DEFAULT_LINKER-SAME: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// DEFAULT_LINKER-SAME: "-o" "a.out"
// DEFAULT_LINKER-SAME: "-z" "pack-relative-relocs"
// DEFAULT_LINKER-SAME: "[[SYSROOT]]/usr/lib/crt0.o"
// DEFAULT_LINKER-SAME: "[[RESOURCE:[^"]+]]/lib/[[TRIPLE]]/clang_rt.crtbegin.o"
// DEFAULT_LINKER-SAME: "[[RESOURCE]]/lib/[[TRIPLE]]/libclang_rt.builtins.a"
// DEFAULT_LINKER-SAME: "-lc" "[[RESOURCE]]/lib/[[TRIPLE]]/clang_rt.crtend.o"

/// Check if the sysroot is passed to the linker.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=TestSysroot \
// RUN:   -static-pie 2>&1 | FileCheck %s --check-prefix=LINKER_SYSROOT
// LINKER_SYSROOT: ld.lld"
// LINKER_SYSROOT-SAME: "--sysroot=TestSysroot"

/// Check that we find crt*.o files in the sysroot, use -static-pie to
/// be able to check for crtbeginS.o and crtendS.o too.
// RUN: %clang -### %s --target=x86_64-unknown-serenity \
// RUN:   --sysroot=%S/Inputs/serenity_tree \
// RUN:   -static-pie 2>&1 | FileCheck %s --check-prefix=CRT_SYSROOT
// CRT_SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CRT_SYSROOT: "[[SYSROOT]]/usr/lib/crt0.o"
// CRT_SYSROOT-SAME: "[[SYSROOT]]/usr/lib/crtbeginS.o"
// CRT_SYSROOT-SAME: "[[SYSROOT]]/usr/lib/crtendS.o"

/// -static-pie suppresses -dynamic-linker.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -static-pie 2>&1 | FileCheck %s --check-prefix=STATIC_PIE
// STATIC_PIE: "-static" "-pie"
// STATIC_PIE-NOT: "-dynamic-linker"
// STATIC_PIE-SAME: "--no-dynamic-linker" "-z" "text"
// STATIC_PIE-SAME: "--eh-frame-hdr"
// STATIC_PIE-SAME: "-z" "pack-relative-relocs"
// STATIC_PIE-SAME: "crt0.o" "crtbeginS.o"
// STATIC_PIE-SAME: "-lc" "crtendS.o"

/// -shared forces use of shared crt files.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -shared 2>&1 | FileCheck %s --check-prefix=SHARED
// SHARED: "-shared"
// SHARED-SAME: "--eh-frame-hdr"
// SHARED-SAME: "-z" "pack-relative-relocs"
// SHARED-NOT: "crt0.o"
// SHARED-SAME: "crtbeginS.o"
// SHARED-SAME: "-lc" "crtendS.o"

/// -static forces use of static crt files.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -static 2>&1 | FileCheck %s --check-prefix=STATIC
// STATIC: "-static"
// STATIC-SAME: "--eh-frame-hdr"
// STATIC-SAME: "-z" "pack-relative-relocs"
// STATIC-SAME: "crt0.o" "crtbeginS.o"
// STATIC-SAME: "-lc" "crtendS.o"

/// -rdynamic passes -export-dynamic.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -rdynamic 2>&1 | FileCheck %s --check-prefix=RDYNAMIC,RDYNAMIC_SHARED
// RDYNAMIC: "-export-dynamic" "-pie"

// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -no-pie -rdynamic 2>&1 | FileCheck %s --check-prefix=RDYNAMIC_NOPIE,RDYNAMIC_SHARED
// RDYNAMIC_NOPIE: "-export-dynamic"
// RDYNAMIC_NOPIE-NOT: "-pie"

// RDYNAMIC_SHARED-SAME: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// RDYNAMIC_SHARED-SAME: "-o" "a.out"
// RDYNAMIC_SHARED-SAME: "-z" "pack-relative-relocs"
// RDYNAMIC_SHARED-SAME: "crt0.o" "crtbeginS.o"
// RDYNAMIC_SHARED-SAME: "-lc" "crtendS.o"

/// -nostdlib suppresses default -l and crt*.o.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir= -nostdlib --rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOSTDLIB
// NOSTDLIB:      "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDLIB-NOT:  crt{{[^./]+}}.o
// NOSTDLIB:      "-L
// NOSTDLIB-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOSTDLIB-NOT:  "-l
// NOSTDLIB-NOT:  libclang_rt.builtins
// NOSTDLIB-NOT:  crt{{[^./]+}}.o

/// -nostartfiles suppresses crt*.o, but not default -l.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir -nostartfiles --rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOSTARTFILES
// NOSTARTFILES:      "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTARTFILES-SAME: {{^}}
// NOSTARTFILES-NOT:  crt{{[^./]+}}.o
// NOSTARTFILES:      "-L
// NOSTARTFILES-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOSTARTFILES:      lib/x86_64-unknown-serenity/libclang_rt.builtins.a"
// NOSTARTFILES:      "-lc"
// NOSTARTFILES-NOT:  crt{{[^./]+}}.o

/// -r suppresses -dynamic-linker, default -l, and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir -r --rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE-NOT:  "-dynamic-linker"
// RELOCATABLE:      "-isysroot" "[[SYSROOT:[^"]+]]"
// RELOCATABLE:      "-internal-isystem"
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
// RELOCATABLE:      "-L
// RELOCATABLE-SAME: {{^}}[[SYSROOT]]/usr/lib"
// RELOCATABLE-NOT:  "-l
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
// RELOCATABLE-NOT:  libclang_rt.builtins

/// -nolibc suppresses -lc but not other default -l.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir -nolibc --rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOLIBC
// NOLIBC:      "-isysroot" "[[SYSROOT:[^"]+]]"
// NOLIBC:      "-internal-isystem"
// NOLIBC:      "[[SYSROOT]]/usr/lib/crt0.o" "[[SYSROOT]]/usr/lib/crtbeginS.o"
// NOLIBC:      "-L
// NOLIBC-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOLIBC-NOT:  "-lc"
// NOLIBC:      "[[RESOURCE:[^"]+]]/lib/x86_64-unknown-serenity/libclang_rt.builtins.a"
// NOLIBC:      "[[SYSROOT]]/usr/lib/crtendS.o"

/// -fsanitize=undefined redirects to Serenity-custom UBSAN runtime.
// RUN: %clang -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/serenity_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir -fsanitize=undefined --rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck %s --check-prefix=UBSAN
// UBSAN-NOT: "libclang_rt.ubsan"
// UBSAN:     "-lubsan"

/// Support for KASAN.
// RUN: %clang -target x86_64-unknown-serenity -fsanitize=kernel-address -### %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=KASAN
// KASAN:      "-fsanitize=kernel-address"
// KASAN-SAME: "-fsanitize-recover=kernel-address"

/// Check C++ stdlib behavior.
// RUN: %clangxx -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   2>&1 | FileCheck %s --check-prefix=DEFAULT_LIBCXX
// DEFAULT_LIBCXX: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// DEFAULT_LIBCXX: "-z" "pack-relative-relocs"
// DEFAULT_LIBCXX: "crt0.o" "crtbeginS.o"
// DEFAULT_LIBCXX: "--push-state"
// DEFAULT_LIBCXX: "--as-needed"
// DEFAULT_LIBCXX: "-lc++"
// DEFAULT_LIBCXX: "--pop-state"
// DEFAULT_LIBCXX: "-lc" "crtendS.o"

// RUN: %clangxx -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -static 2>&1 | FileCheck %s --check-prefix=STATIC_LIBCXX
// STATIC_LIBCXX: "-z" "pack-relative-relocs"
// STATIC_LIBCXX: "crt0.o" "crtbeginS.o"
// STATIC_LIBCXX: "--push-state"
// STATIC_LIBCXX: "--as-needed"
// STATIC_LIBCXX: "-lc++"
// STATIC_LIBCXX: "--pop-state"
// STATIC_LIBCXX: "-lc" "crtendS.o"

// RUN: %clangxx -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -static-libstdc++ 2>&1 | FileCheck %s --check-prefix=STATIC_LIBSTDCXX
// STATIC_LIBSTDCXX: "-z" "pack-relative-relocs"
// STATIC_LIBSTDCXX: "crt0.o" "crtbeginS.o"
// STATIC_LIBSTDCXX: "--push-state"
// STATIC_LIBSTDCXX: "--as-needed"
// STATIC_LIBSTDCXX: "-Bstatic"
// STATIC_LIBSTDCXX: "-lc++"
// STATIC_LIBSTDCXX: "-Bdynamic"
// STATIC_LIBSTDCXX: "--pop-state"
// STATIC_LIBSTDCXX: "-lc" "crtendS.o"

// RUN: %clangxx -### %s --target=x86_64-unknown-serenity --sysroot=%S/Inputs/empty_tree \
// RUN:   -nostdlib++ 2>&1 | FileCheck %s --check-prefix=NO_LIBCXX
// NO_LIBCXX: "-z" "pack-relative-relocs"
// NO_LIBCXX: "crt0.o" "crtbeginS.o"
// NO_LIBCXX-NOT: "-lc++"
// NO_LIBCXX-SAME: "-lc" "crtendS.o"

/// Check that unwind tables are enabled.
// RUN: %clang --target=x86_64-unknown-serenity -### -S %s 2>&1 | \
// RUN:   FileCheck -check-prefix=UNWIND-TABLES %s
// RUN: %clang --target=aarch64-unknown-serenity -### -S %s 2>&1 | \
// RUN:   FileCheck -check-prefix=UNWIND-TABLES %s
// RUN: %clang --target=riscv64-unknown-serenity -### -S %s 2>&1 | \
// RUN:   FileCheck -check-prefix=UNWIND-TABLES %s
// UNWIND-TABLES: "-funwind-tables=2"

/// Check that parameters are forwarded to the linker.
// RUN: %clang --target=x86_64-unknown-serenity -### %s -L/foo -u bar -T script.ld -s -t -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK
// LINK: ld.lld"
// LINK-SAME: "-L/foo"
// LINK-SAME: "-u" "bar"
// LINK-SAME: "-T" "script.ld"
// LINK-SAME: "-s"
// LINK-SAME: "-t"
// LINK-SAME: "-r"

// RUN: %clang --target=x86_64-unknown-serenity -### %s -Wl,--compress-debug-sections=zlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COMPRESS
// COMPRESS: ld.lld"
// COMPRESS: "--compress-debug-sections=zlib"

/// Check LTO.
// RUN: %clang --target=x86_64-unknown-serenity -flto %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LTO_FULL
// LTO_FULL: "-plugin-opt=
// LTO_FULL-NOT: "-plugin-opt=thinlto"

// RUN: %clang --target=x86_64-unknown-serenity -flto=thin %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LTO_THIN
// LTO_THIN: "-plugin-opt=thinlto"
