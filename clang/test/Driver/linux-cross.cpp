// UNSUPPORTED: system-windows

/// Test native x86-64 in the tree.
// RUN: %clang -### %s --target=x86_64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64
// DEBIAN_X86_64:      "-internal-isystem"
// DEBIAN_X86_64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10/backward"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../x86_64-linux-gnu/include"
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
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64_M32
// DEBIAN_X86_64_M32:      "-internal-isystem"
// DEBIAN_X86_64_M32-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10/32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10/backward"
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
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/lib"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test a cross compiler.
// RUN: %clang -### %s --target=aarch64-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=DEBIAN_AARCH64
// DEBIAN_AARCH64:      "-internal-isystem"
// DEBIAN_AARCH64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10/aarch64-linux-gnu"
// DEBIAN_AARCH64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc-cross/aarch64-linux-gnu/10/../../../../aarch64-linux-gnu/include/c++/10/backward"
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

/// LDSO_ARCH is i386 for all x86-32 variants.
// RUN: %clang -### %s --target=i686-linux-musl --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=MUSL_I686
// MUSL_I686: "-dynamic-linker" "/lib/ld-musl-i386.so.1"

// RUN: %clang -### %s --target=x86_64-linux-muslx32 --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform 2>&1 | FileCheck %s --check-prefix=MUSL_X32
// MUSL_X32: "-dynamic-linker" "/lib/ld-musl-x32.so.1"
