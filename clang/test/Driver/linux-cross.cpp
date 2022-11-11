// UNSUPPORTED: system-windows

/// Test native GCC installation on Arch Linux i686.
// RUN: %clang -### %s --target=i686-linux-gnu --sysroot=%S/Inputs/archlinux_i686_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=ARCH_I686
// ARCH_I686:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// ARCH_I686:      "-internal-isystem"
// ARCH_I686-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../include/c++/11.1.0"
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../include/c++/11.1.0/i686-pc-linux-gnu"
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../include/c++/11.1.0/backward"
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
/// This resolves to /usr/i686-linux-gnu/include. Because it does not exist,
/// having it does no harm albeit not ideal.
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../i686-pc-linux-gnu/include"
// ARCH_I686:      "-internal-externc-isystem"
// ARCH_I686-SAME: {{^}} "[[SYSROOT]]/include"
// ARCH_I686-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// ARCH_I686:      "-L
// ARCH_I686-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0"
// ARCH_I686-SAME: {{^}} "-L[[SYSROOT]]/lib"
// ARCH_I686-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test native x86-64 in the tree.
// RUN: %clang -### %s --target=x86_64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64
// DEBIAN_X86_64:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_X86_64:      "-internal-isystem"
// DEBIAN_X86_64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10/backward"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/include"
/// We set explicit -ccc-install-dir ensure that Clang does not pick up extra
/// library directories which may be present in the runtimes build.
// DEBIAN_X86_64:      "-L
// DEBIAN_X86_64-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10"
/// Debian patches MULTILIB_OSDIRNAMES (../lib64 -> ../lib), so gcc uses 'lib' instead of 'lib64'.
/// This difference does not matter in practice.
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib64"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/lib/x86_64-linux-gnu"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib64"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/x86_64-linux-gnu"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib64"
/// /usr/x86_64-linux-gnu does not exist, so there is no /usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/lib.
/// -ccc-install-dir is not within sysroot. No bin/../lib.
/// $sysroot/lib and $sysroot/usr/lib. Fallback when GCC installation is unavailable.
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test -m32.
// RUN: %clang -### %s --target=x86_64-linux-gnu -m32 --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64_M32
// DEBIAN_X86_64_M32:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_X86_64_M32:      "-internal-isystem"
// DEBIAN_X86_64_M32-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10/32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10/backward"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/include"
// DEBIAN_X86_64_M32:      "-internal-externc-isystem"
// DEBIAN_X86_64_M32-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-linux-gnu"
// DEBIAN_X86_64_M32:      "-L
// DEBIAN_X86_64_M32-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-linux-gnu"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-linux-gnu"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test native GCC installation on Debian i386.
// RUN: %clang -### %s --target=i686-linux-gnu --sysroot=%S/Inputs/debian_i386_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_I686
// DEBIAN_I686:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_I686:      "-internal-isystem"
// DEBIAN_I686-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/c++/10"
/// Debian specific - the path component after 'include' is i386-linux-gnu even
/// though the installation is i686-linux-gnu.
// DEBIAN_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/i386-linux-gnu/c++/10"
// DEBIAN_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/c++/10/backward"
// DEBIAN_I686-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
/// This resolves to /usr/i686-linux-gnu/include. Because it does not exist,
/// having it does no harm albeit not ideal.
// DEBIAN_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../i686-linux-gnu/include"
// DEBIAN_I686:      "-internal-externc-isystem"
// DEBIAN_I686-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-linux-gnu"
// DEBIAN_I686:      "-L
// DEBIAN_I686-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/10"
// DEBIAN_I686-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-linux-gnu"
// DEBIAN_I686-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-linux-gnu"
// DEBIAN_I686-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_I686-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test -m64 on Debian i386.
// RUN: %clang -### %s --target=i686-linux-gnu --sysroot=%S/Inputs/debian_i386_tree -m64 \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_I686_M64
// DEBIAN_I686_M64:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_I686_M64:      "-internal-isystem"
// DEBIAN_I686_M64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/c++/10"
/// Debian specific - the path component after 'include' is i386-linux-gnu even
/// though the installation is i686-linux-gnu.
// DEBIAN_I686_M64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/i386-linux-gnu/c++/10/64"
// DEBIAN_I686_M64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../include/c++/10/backward"
// DEBIAN_I686_M64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_I686_M64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// DEBIAN_I686_M64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-linux-gnu/10/../../../../i686-linux-gnu/include"
// DEBIAN_I686_M64:      "-internal-externc-isystem"
// DEBIAN_I686_M64-SAME: {{^}} "[[SYSROOT]]/usr/include/x86_64-linux-gnu"
// DEBIAN_I686_M64:      "-L
// DEBIAN_I686_M64-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/10/64"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/10/../../../../lib64"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/lib/x86_64-linux-gnu"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib64"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/x86_64-linux-gnu"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib64"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_I686_M64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test a cross compiler.
// RUN: %clang -### %s --target=aarch64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_AARCH64
// DEBIAN_AARCH64:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_AARCH64:      "-internal-isystem"
// DEBIAN_AARCH64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10/aarch64-linux-gnu"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10/backward"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include"
// DEBIAN_AARCH64:      "-L
// DEBIAN_AARCH64-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc-cross/aarch64-linux-gnu/10"
/// Debian patches MULTILIB_OSDIRNAMES (../lib64 -> ../lib), so aarch64-linux-gnu-gcc uses 'lib' instead of 'lib64'.
/// This difference does not matter in practice.
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../lib64"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/lib/aarch64-linux-gnu"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib64"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/aarch64-linux-gnu"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib64"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/lib"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_AARCH64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test native x86-64 with -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=on.
/// FIXME -internal-isystem .*bin/../include/x86_64-linux-gnu/c++/v1 and -L[[PREFIX]]/bin/../lib/x86_64-linux-gnu are missing.
// RUN: %clang -### %s --target=x86_64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/debian_per_target_tree/usr/lib/llvm-14/bin -resource-dir=%S/Inputs/debian_per_target_tree/usr/lib/llvm-14/lib/clang/14.0.0 \
// RUN:   --stdlib=libc++ --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64_PER_TARGET
// DEBIAN_X86_64_PER_TARGET:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_X86_64_PER_TARGET:      "-internal-isystem"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "[[PREFIX:[^"]+llvm-14]]/bin/../include/c++/v1"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/local/include"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/include"

// DEBIAN_X86_64_PER_TARGET:      "-L
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10"
/// Debian patches MULTILIB_OSDIRNAMES (../lib64 -> ../lib), so gcc uses 'lib' instead of 'lib64'.
/// This difference does not matter in practice.
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib64"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib/x86_64-linux-gnu"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib64"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/x86_64-linux-gnu"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib64"
/// /usr/x86_64-linux-gnu does not exist, so there is no /usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/lib.
/// -ccc-install-dir is not within sysroot. No bin/../lib.
/// $sysroot/lib and $sysroot/usr/lib. Fallback when GCC installation is unavailable.
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_X86_64_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test -m32.
/// FIXME -internal-isystem .*bin/../include/i386-linux-gnu/c++/v1 and -L[[PREFIX]]/bin/../lib/i386-linux-gnu are missing.
// RUN: %clang -### %s --target=x86_64-linux-gnu -m32 --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/debian_per_target_tree/usr/lib/llvm-14/bin -resource-dir=%S/Inputs/debian_per_target_tree/usr/lib/llvm-14/lib/clang/14.0.0 \
// RUN:   --stdlib=libc++ --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64_M32_PER_TARGET
// DEBIAN_X86_64_M32_PER_TARGET:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// DEBIAN_X86_64_M32_PER_TARGET:      "-internal-isystem"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "[[PREFIX:[^"]+llvm-14]]/bin/../include/c++/v1"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/local/include"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/include"

// DEBIAN_X86_64_M32_PER_TARGET:      "-L
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/32"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib32"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-linux-gnu"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-linux-gnu"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_X86_64_M32_PER_TARGET-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// LDSO_ARCH is i386 for all x86-32 variants.
// RUN: %clang -### %s --target=i686-linux-musl --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=MUSL_I686
// MUSL_I686: "-dynamic-linker" "/lib/ld-musl-i386.so.1"

// RUN: %clang -### %s --target=x86_64-linux-muslx32 --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=MUSL_X32
// RUN: %clang -### %s --target=i686-linux-musl -mx32 --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=MUSL_X32
// MUSL_X32: "-dynamic-linker" "/lib/ld-musl-x32.so.1"

/// -r suppresses -dynamic-linker, default -l, and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=x86_64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform -r 2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE-NOT:  "-dynamic-linker"
// RELOCATABLE:      "-internal-isystem"
// RELOCATABLE-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// RELOCATABLE:      "-L
// RELOCATABLE-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10"
// RELOCATABLE-NOT:  "-l
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
