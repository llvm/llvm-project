// UNSUPPORTED: system-windows
/// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### --target=loongarch32 2>&1 | FileCheck --check-prefix=CC1 %s -DTRIPLE=loongarch32
// RUN: %clang %s -### --target=loongarch64 2>&1 | FileCheck --check-prefix=CC1 %s -DTRIPLE=loongarch64

// CC1: "-cc1" "-triple" "[[TRIPLE]]"

/// In the below tests, --rtlib=platform is used so that the driver ignores
/// the configure-time CLANG_DEFAULT_RTLIB option when choosing the runtime lib.

// RUN: env "PATH=" %clang -### %s -fuse-ld=ld -no-pie -mabi=lp64d \
// RUN:   --target=loongarch64-unknown-linux-gnu --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_loongarch_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_loongarch_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck --check-prefix=LA64 %s

// LA64: "{{.*}}/Inputs/multilib_loongarch_linux_sdk/lib/gcc/loongarch64-unknown-linux-gnu/12.1.0/../../../../loongarch64-unknown-linux-gnu/bin/ld"
// LA64-SAME: {{^}} "--sysroot={{.*}}/Inputs/multilib_loongarch_linux_sdk/sysroot"
// LA64-SAME: "-m" "elf64loongarch"
// LA64-SAME: "-dynamic-linker" "/lib64/ld-linux-loongarch-lp64d.so.1"
// LA64-SAME: "{{.*}}/Inputs/multilib_loongarch_linux_sdk/lib/gcc/loongarch64-unknown-linux-gnu/12.1.0/crtbegin.o"
// LA64-SAME: "-L{{.*}}/Inputs/multilib_loongarch_linux_sdk/lib/gcc/loongarch64-unknown-linux-gnu/12.1.0"
// LA64-SAME: {{^}} "-L{{.*}}/Inputs/multilib_loongarch_linux_sdk/lib/gcc/loongarch64-unknown-linux-gnu/12.1.0/../../../../loongarch64-unknown-linux-gnu/lib/../lib64"
// LA64-SAME: {{^}} "-L{{.*}}/Inputs/multilib_loongarch_linux_sdk/sysroot/usr/lib/../lib64"
// LA64-SAME: {{^}} "-L{{.*}}/Inputs/multilib_loongarch_linux_sdk/lib/gcc/loongarch64-unknown-linux-gnu/12.1.0/../../../../loongarch64-unknown-linux-gnu/lib"
// LA64-SAME: {{^}} "-L{{.*}}/Inputs/multilib_loongarch_linux_sdk/sysroot/usr/lib"
