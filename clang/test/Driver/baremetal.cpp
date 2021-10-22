// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     -T semihosted.lds \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-C %s
// CHECK-V6M-C: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "thumbv6m-none-unknown-eabi"
// CHECK-V6M-C-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-C-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-V6M-C-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECk-V6M-C-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include"
// CHECK-V6M-C-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-V6M-C-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-C-SAME: "-T" "semihosted.lds" "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-V6M-C-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-V6M-C-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-C-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-C-SAME: "-o" "{{.*}}.o"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     -nostdlibinc -nobuiltininc \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBINC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     -nostdinc \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBINC %s
// CHECK-V6M-LIBINC-NOT: "-internal-isystem"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-DEFAULTCXX %s
// CHECK-V6M-DEFAULTCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-DEFAULTCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-V6M-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBCXX %s
// CHECK-V6M-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-V6M-LIBCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-V6M-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-LIBCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-V6M-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBSTDCXX %s
// CHECK-V6M-LIBSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-V6M-LIBSTDCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}6.0.0"
// CHECK-V6M-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-LIBSTDCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-LIBSTDCXX-SAME: "-lstdc++" "-lsupc++" "-lunwind"
// CHECK-V6M-LIBSTDCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-LIBSTDCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -nodefaultlibs \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-NDL %s
// CHECK-V6M-NDL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-NDL: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-NDL-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-NDL-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -target arm-none-eabi -v 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-THREAD-MODEL
// CHECK-THREAD-MODEL: Thread model: posix

// RUN: %clangxx -target arm-none-eabi -mthread-model single -v 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-THREAD-MODEL-SINGLE
// CHECK-THREAD-MODEL-SINGLE: Thread model: single

// RUN: %clangxx -target arm-none-eabi -mthread-model posix -v 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-THREAD-MODEL-POSIX
// CHECK-THREAD-MODEL-POSIX: Thread model: posix

// RUN: %clang -### -target arm-none-eabi -rtlib=libgcc -v %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-RTLIB-GCC
// CHECK-RTLIB-GCC: -lgcc

// RUN: %clang -### -target arm-none-eabi -v %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SYSROOT-INC
// CHECK-SYSROOT-INC-NOT: "-internal-isystem" "include"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:      -target aarch64-none-elf \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64-NO-HOST-INC %s
// Verify that the bare metal driver does not include any host system paths:
// CHECK-AARCH64-NO-HOST-INC: InstalledDir: [[INSTALLEDDIR:.+]]
// CHECK-AARCH64-NO-HOST-INC: "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK-AARCH64-NO-HOST-INC-SAME: "-internal-isystem" "[[INSTALLEDDIR]]/../lib/clang-runtimes/aarch64-none-elf/include/c++/v1"
// CHECK-AARCH64-NO-HOST-INC-SAME: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-AARCH64-NO-HOST-INC-SAME: "-internal-isystem" "[[INSTALLEDDIR]]/../lib/clang-runtimes/aarch64-none-elf/include"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64 %s
// CHECK-RV64: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv64-unknown-unknown-elf"
// CHECK-RV64-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV64-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECk-RV64-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include"
// CHECK-RV64-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV64-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-RV64-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-RV64-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-DEFAULTCXX %s
// CHECK-RV64-DEFAULTCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-DEFAULTCXX-SAME: "-L[[RESOURCE_DIR]]{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV64-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBCXX %s
// CHECK-RV64-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-RV64-LIBCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV64-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-LIBCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV64-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBSTDCXX %s
// CHECK-RV64-LIBSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV64-LIBSTDCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}8.0.1"
// CHECK-RV64-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-LIBSTDCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-LIBSTDCXX-SAME: "-lstdc++" "-lsupc++" "-lunwind"
// CHECK-RV64-LIBSTDCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-LIBSTDCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32 %s
// CHECK-RV32: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv32-unknown-unknown-elf"
// CHECK-RV32-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV32-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}include"
// CHECK-RV32-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV32-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-RV32-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-RV32-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-DEFAULTCXX %s
// CHECK-RV32-DEFAULTCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
// CHECK-RV32-DEFAULTCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV32-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-LIBCXX %s
// CHECK-RV32-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-RV32-LIBCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
// CHECK-RV32-LIBCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV32-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-LIBSTDCXX %s
// CHECK-RV32-LIBSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32-LIBSTDCXX-SAME: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}8.0.1"
// CHECK-RV32-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
// CHECK-RV32-LIBSTDCXX-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-LIBSTDCXX-SAME: "-lstdc++" "-lsupc++" "-lunwind"
// CHECK-RV32-LIBSTDCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-LIBSTDCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     -nostdlibinc -nobuiltininc \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBINC %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     -nostdinc \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBINC %s
// CHECK-RV64-LIBINC-NOT: "-internal-isystem"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:     -nodefaultlibs \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-NDL %s
// CHECK-RV64-NDL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64-NDL: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-NDL-SAME: "-L[[RESOURCE_DIR]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-NDL-SAME: "-o" "{{.*}}.o"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     -march=rv64imafdc -mabi=lp64d \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64FD %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     -march=rv64gc -mabi=lp64d \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64FD %s

// CHECK-RV64FD: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv64-unknown-unknown-elf"
// CHECK-RV64FD-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV64FD-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV64FD-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv64imafdc{{[/\\]+}}lp64d{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECk-RV64FD-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv64imafdc{{[/\\]+}}lp64d{{[/\\]+}}include"
// CHECK-RV64FD-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV64FD-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64FD-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}rv64imafdc{{[/\\]+}}lp64d{{[/\\]+}}lib"
// CHECK-RV64FD-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal{{[/\\]+}}rv64imafdc{{[/\\]+}}lp64d"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32i -mabi=ilp32 \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32I %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32ic -mabi=ilp32 \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32I %s

// CHECK-RV32I: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv32-unknown-unknown-elf"
// CHECK-RV32I-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32I-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV32I-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32i{{[/\\]+}}ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32I-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32i{{[/\\]+}}ilp32{{[/\\]+}}include"
// CHECK-RV32I-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV32I-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32I-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}rv32i{{[/\\]+}}ilp32{{[/\\]+}}lib"
// CHECK-RV32I-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal{{[/\\]+}}rv32i{{[/\\]+}}ilp32"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32im -mabi=ilp32 \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IM %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32imc -mabi=ilp32 \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IM %s

// CHECK-RV32IM: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv32-unknown-unknown-elf"
// CHECK-RV32IM-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32IM-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV32IM-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32im{{[/\\]+}}ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32IM-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32im{{[/\\]+}}ilp32{{[/\\]+}}include"
// CHECK-RV32IM-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV32IM-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32IM-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}rv32im{{[/\\]+}}ilp32{{[/\\]+}}lib"
// CHECK-RV32IM-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal{{[/\\]+}}rv32im{{[/\\]+}}ilp32"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32iac -mabi=ilp32 \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IAC %s

// CHECK-RV32IAC: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv32-unknown-unknown-elf"
// CHECK-RV32IAC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32IAC-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV32IAC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32iac{{[/\\]+}}ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32IAC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32iac{{[/\\]+}}ilp32{{[/\\]+}}include"
// CHECK-RV32IAC-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV32IAC-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32IAC-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}rv32iac{{[/\\]+}}ilp32{{[/\\]+}}lib"
// CHECK-RV32IAC-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal{{[/\\]+}}rv32iac{{[/\\]+}}ilp32"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32imafc -mabi=ilp32f \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IMAFC %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32imafdc -mabi=ilp32f \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IMAFC %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     -march=rv32gc -mabi=ilp32f \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32IMAFC %s

// CHECK-RV32IMAFC: "[[PREFIX_DIR:.*]]{{[/\\]+}}{{[^/^\\]+}}{{[/\\]+}}clang{{.*}}" "-cc1" "-triple" "riscv32-unknown-unknown-elf"
// CHECK-RV32IMAFC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-RV32IMAFC-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-RV32IMAFC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32imafc{{[/\\]+}}ilp32f{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32IMAFC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}rv32imafc{{[/\\]+}}ilp32f{{[/\\]+}}include"
// CHECK-RV32IMAFC-SAME: "-x" "c++" "{{.*}}baremetal.cpp"
// CHECK-RV32IMAFC-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32IMAFC-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}rv32imafc{{[/\\]+}}ilp32f{{[/\\]+}}lib"
// CHECK-RV32IMAFC-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal{{[/\\]+}}rv32imafc{{[/\\]+}}ilp32f"
