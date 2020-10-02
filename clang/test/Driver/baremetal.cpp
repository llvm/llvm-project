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
// CHECK-V6M-LIBSTDCXX-SAME: "-lstdc++" "-lsupc++" "-lunwind"
// CHECK-V6M-LIBSTDCXX-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-LIBSTDCXX-SAME: "-o" "{{.*}}.o"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:     --sysroot=%S/Inputs/baremetal_arm \
// RUN:     -nodefaultlibs \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-NDL %s
// CHECK-V6M-NDL: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-NDL-SAME: "-L{{[^"]*}}{{[/\\]+}}lib{{(64)?}}{{[/\\]+}}clang{{[/\\]+}}{{.*}}{{[/\\]+}}lib{{[/\\]+}}baremetal" "-o" "{{.*}}.o"

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
