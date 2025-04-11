// UNSUPPORTED: system-windows

/// Test native GCC installation on Arch Linux i686.
// RUN: %clang -### %s --target=i686-linux-gnu --sysroot=%S/Inputs/archlinux_i686_tree -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:   --stdlib=platform --rtlib=platform --unwindlib=platform \
// RUN:   --gcc-install-dir=%S/Inputs/archlinux_i686_tree/usr/lib/gcc/i686-pc-linux-gnu/11.1.0 2>&1 | FileCheck %s --check-prefix=ARCH_I686
// ARCH_I686:      "-internal-isystem"
// ARCH_I686-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../include/c++/11.1.0"
// ARCH_I686-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0/../../../../include/c++/11.1.0/i686-pc-linux-gnu"
// ARCH_I686:      "-L
// ARCH_I686-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-pc-linux-gnu/11.1.0"
// ARCH_I686-SAME: {{^}} "-L[[SYSROOT]]/lib"
// ARCH_I686-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

/// Test native GCC installation on Debian amd64. --gcc-install-dir= may end with /.
// RUN: %clangxx %s -### --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir --stdlib=platform --rtlib=platform --unwindlib=platform \
// RUN:   --gcc-install-dir=%S/Inputs/debian_multiarch_tree/usr/lib/gcc/x86_64-linux-gnu/10/ 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64
// DEBIAN_X86_64:      "-internal-isystem"
// DEBIAN_X86_64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10"
// DEBIAN_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10/backward"
/// We set explicit -ccc-install-dir ensure that Clang does not pick up extra
/// library directories which may be present in the runtimes build.
// DEBIAN_X86_64:      "-L
// DEBIAN_X86_64-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10"
// DEBIAN_X86_64-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib64"

/// Test -m32.
// RUN: %clangxx %s -### --target=x86_64-unknown-linux-gnu -m32 --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir --stdlib=platform --rtlib=platform --unwindlib=platform \
// RUN:   --gcc-install-dir=%S/Inputs/debian_multiarch_tree/usr/lib/gcc/x86_64-linux-gnu/10/ 2>&1 | FileCheck %s --check-prefix=DEBIAN_X86_64_M32
// DEBIAN_X86_64_M32:      "-internal-isystem"
// DEBIAN_X86_64_M32-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/c++/10"
// DEBIAN_X86_64_M32-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../include/x86_64-linux-gnu/c++/10/32"
// DEBIAN_X86_64_M32:      "-L
// DEBIAN_X86_64_M32-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/32"
// DEBIAN_X86_64_M32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib32"

/// Test GCC installation on bare-metal RISCV64.
// RUN: %clang -### %s --target=riscv64-unknown-elf --sysroot=%S/Inputs/multilib_riscv_elf_sdk/riscv64-unknown-elf/ --stdlib=platform --rtlib=platform \
// RUN:   --gcc-install-dir=%S/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/ 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ELF_RISCV64
// ELF_RISCV64:      "-internal-isystem"
// ELF_RISCV64-SAME: {{^}} "[[SYSROOT:[^"]+]]/include/c++/8.2.0"
// ELF_RISCV64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/include/c++/8.2.0/riscv64-unknown-elf/rv64imac/lp64"
// ELF_RISCV64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/include/c++/8.2.0/backward"
// ELF_RISCV64:      "-L
// ELF_RISCV64-SAME: {{^}}[[SYSROOT:[^"]+]]/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imac/lp64"
// ELF_RISCV64-SAME: {{^}} "-L[[SYSROOT]]/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/lib/rv64imac/lp64"

// RUN: not %clangxx %s -### --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin -resource-dir=%S/Inputs/resource_dir --stdlib=platform --rtlib=platform \
// RUN:   --gcc-install-dir=%S/Inputs/debian_multiarch_tree/usr/lib/gcc/x86_64-linux-gnu 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: error: '{{.*}}/usr/lib/gcc/x86_64-linux-gnu' does not contain a GCC installation
