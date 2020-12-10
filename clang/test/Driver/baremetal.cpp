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
// CHECK-V6M-C-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-C-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-V6M-C-SAME: "-T" "semihosted.lds" "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
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
// CHECK-V6M-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-V6M-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBCXX %s
// CHECK-V6M-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-V6M-LIBCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-V6M-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-V6M-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-LIBSTDCXX %s
// CHECK-V6M-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-V6M-LIBSTDCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}6.0.0"
// CHECK-V6M-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib"
// CHECK-V6M-LIBSTDCXX-SAME: "-lstdc++" "-lsupc++" "-lunwind"
// CHECK-V6M-LIBSTDCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-LIBSTDCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -nodefaultlibs \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-NDL %s
// CHECK-V6M-NDL: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-V6M-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}baremetal_arm{{[/\\]+}}lib" "-o" "{{.*}}.o"

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
// CHECK-RV64-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-RV64-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-RV64-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-DEFAULTCXX %s
// CHECK-RV64-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV64-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBCXX %s
// CHECK-RV64-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-RV64-LIBCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV64-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV64-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv64"
// CHECK-RV64-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv64-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-LIBSTDCXX %s
// CHECK-RV64-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV64-LIBSTDCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}8.0.1"
// CHECK-RV64-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
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
// CHECK-RV32-SAME: "-L[[RESOURCE_DIR:[^"]+]]{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-SAME: "-L[[SYSROOT:[^"]+]]{{[/\\]+}}lib"
// CHECK-RV32-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-RV32-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-DEFAULTCXX %s
// CHECK-RV32-DEFAULTCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-DEFAULTCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
// CHECK-RV32-DEFAULTCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV32-DEFAULTCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-DEFAULTCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-LIBCXX %s
// CHECK-RV32-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-RV32-LIBCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32-LIBCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-LIBCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
// CHECK-RV32-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-RV32-LIBCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-riscv32"
// CHECK-RV32-LIBCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-unknown-elf \
// RUN:     --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:   | FileCheck --check-prefix=CHECK-RV32-LIBSTDCXX %s
// CHECK-RV32-LIBSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-RV32-LIBSTDCXX: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}8.0.1"
// CHECK-RV32-LIBSTDCXX: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV32-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV32-LIBSTDCXX-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv32_tree{{[/\\]+}}riscv32-unknown-elf{{[/\\]+}}lib"
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
// CHECK-RV64-NDL: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-RV64-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal"
// CHECK-RV64-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}Inputs{{[/\\]+}}basic_riscv64_tree{{[/\\]+}}riscv64-unknown-elf{{[/\\]+}}lib"
// CHECK-RV64-NDL-SAME: "-o" "{{.*}}.o"
