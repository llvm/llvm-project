// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-linux -fc++-abi=itanium -fsanitize=address | FileCheck --check-prefix=CHECK-LINUX %s
// CHECK-LINUX: --target=aarch64-unknown-linux

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-fuchsia -fsanitize=hwaddress | FileCheck --check-prefix=CHECK-FUCHSIA %s
// CHECK-FUCHSIA: --target=aarch64-unknown-fuchsia

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=soft -fno-exceptions -fno-rtti | FileCheck --check-prefix=CHECK-ARMV4T %s
// CHECK-ARMV4T: --target=armv4t-unknown-none-eabi
// CHECK-ARMV4T: -mfloat-abi=soft
// CHECK-ARMV4T: -mfpu=none

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=armv7em-none-eabi -mfloat-abi=softfp | FileCheck --check-prefix=CHECK-SOFTFP %s
// CHECK-SOFTFP: --target=thumbv7em-unknown-none-eabi
// CHECK-SOFTFP: -mfloat-abi=softfp
// CHECK-SOFTFP: -mfpu=fpv4-sp-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabihf -march=armv7em -mfpu=fpv5-d16 | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD: --target=thumbv7em-unknown-none-eabihf
// CHECK-HARD: -mfloat-abi=hard
// CHECK-HARD: -mfpu=fpv5-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -mabi=aapcs | FileCheck --check-prefix=CHECK-ABI-AAPCS %s
// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -mabi=aapcs-soft | FileCheck --check-prefix=CHECK-ABI-AAPCS-SOFT %s
// CHECK-ABI-AAPCS: -mabi=aapcs
// CHECK-ABI-AAPCS-SOFT: -mabi=aapcs-soft

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=soft -march=armv8-m.main+nofp | FileCheck --check-prefix=CHECK-V8MMAIN-NOFP %s
// CHECK-V8MMAIN-NOFP: --target=thumbv8m.main-unknown-none-eabi
// CHECK-V8MMAIN-NOFP: -mfloat-abi=soft
// CHECK-V8MMAIN-NOFP: -mfpu=none

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=hard -march=armv8.1m.main+mve.fp | FileCheck --check-prefix=CHECK-MVE %s
// CHECK-MVE: --target=thumbv8.1m.main-unknown-none-eabihf
// CHECK-MVE: -march=thumbv8.1m.main{{.*}}+mve{{.*}}+mve.fp{{.*}}
// CHECK-MVE: -mfloat-abi=hard
// CHECK-MVE: -mfpu=fp-armv8-fullfp16-sp-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -march=armv8.1m.main+mve+nofp | FileCheck --check-prefix=CHECK-MVENOFP %s
// CHECK-MVENOFP: -march=thumbv8.1m.main{{.*}}+mve{{.*}}
// CHECK-MVENOFP-NOT: -march=thumbv8.1m.main{{.*}}+mve.fp{{.*}}
// CHECK-MVENOFP: -mfpu=none

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabihf -march=armv8.1m.main+fp.dp | FileCheck --check-prefix=CHECK-V8_1_FP_DP %s
// CHECK-V8_1_FP_DP: -march=thumbv8.1m.main{{.*}}
// CHECK-V8_1_FP_DP: -mfloat-abi=hard
// CHECK-V8_1_FP_DP: -mfpu=fp-armv8-fullfp16-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabihf -march=armv8.1m.main+nofp+fp+nofp.dp | FileCheck --check-prefix=CHECK-V8_1_NO_FP_DP %s
// CHECK-V8_1_NO_FP_DP: -march=thumbv8.1m.main{{.*}}
// CHECK-V8_1_NO_FP_DP: -mfloat-abi=hard
// CHECK-V8_1_NO_FP_DP: -mfpu=fp-armv8-fullfp16-sp-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabihf -mcpu=cortex-m85+nofp.dp | FileCheck --check-prefix=CHECK-M85_NO_FP_DP %s
// CHECK-M85_NO_FP_DP: -march=thumbv8.1m.main{{.*}}
// CHECK-M85_NO_FP_DP: -mfloat-abi=hard
// CHECK-M85_NO_FP_DP: -mfpu=fp-armv8-fullfp16-sp-d16

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -march=armv8-a+lse | FileCheck --check-prefix=CHECK-LSE %s
// CHECK-LSE: --target=aarch64-unknown-none-elf
// CHECK-LSE: -march=armv8-a{{.*}}+lse{{.*}}

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -march=armv8.5-a+sve+sve2 | FileCheck --check-prefix=CHECK-SVE2 %s
// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -march=armv9-a            | FileCheck --check-prefix=CHECK-SVE2 %s
// CHECK-SVE2: --target=aarch64-unknown-none-elf
// CHECK-SVE2: -march=armv{{.*}}-a{{.*}}+simd{{.*}}+sve{{.*}}+sve2{{.*}}

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mbranch-protection=standard    | FileCheck --check-prefix=CHECK-BRANCH-PROTECTION %s
// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -mbranch-protection=standard | FileCheck --check-prefix=CHECK-BRANCH-PROTECTION %s
// CHECK-BRANCH-PROTECTION: -mbranch-protection=standard

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mno-unaligned-access | FileCheck --check-prefix=CHECK-NO-UNALIGNED-ACCESS %s
// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -mno-unaligned-access | FileCheck --check-prefix=CHECK-NO-UNALIGNED-ACCESS %s
// CHECK-NO-UNALIGNED-ACCESS: -mno-unaligned-access

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=arm-none-eabi -mbig-endian | FileCheck --check-prefix=CHECK-BIG-ENDIAN %s
// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=aarch64-none-elf -mbig-endian | FileCheck --check-prefix=CHECK-BIG-ENDIAN %s
// CHECK-BIG-ENDIAN: -mbig-endian

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=riscv32-none-elf -march=rv32g | FileCheck --check-prefix=CHECK-RV32 %s
// CHECK-RV32: --target=riscv32-unknown-none-elf
// CHECK-RV32: -mabi=ilp32d
// CHECK-RV32: -march=rv32i{{[0-9]+p[0-9]+}}_m{{[0-9]+p[0-9]+}}_a{{[0-9]+p[0-9]+}}_f{{[0-9]+p[0-9]+}}_d{{[0-9]+p[0-9]+}}_zicsr{{[0-9]+p[0-9]+}}_zifencei{{[0-9]+p[0-9]+}}_zmmul{{[0-9]+p[0-9]+}}

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=riscv64-none-elf -march=rv64g | FileCheck --check-prefix=CHECK-RV64 %s
// CHECK-RV64: --target=riscv64-unknown-none-elf
// CHECK-RV64: -mabi=lp64d
// CHECK-RV64: -march=rv64i{{[0-9]+p[0-9]+}}_m{{[0-9]+p[0-9]+}}_a{{[0-9]+p[0-9]+}}_f{{[0-9]+p[0-9]+}}_d{{[0-9]+p[0-9]+}}_zicsr{{[0-9]+p[0-9]+}}_zifencei{{[0-9]+p[0-9]+}}_zmmul{{[0-9]+p[0-9]+}}

// RUN: %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml -print-multi-flags-experimental --target=riscv32-none-elf -march=rv32e_zicsr_c | FileCheck --check-prefix=CHECK-RV32E-ORDER %s
// CHECK-RV32E-ORDER: --target=riscv32-unknown-none-elf
// CHECK-RV32E-ORDER: -mabi=ilp32e
// CHECK-RV32E-ORDER: -march=rv32e{{[0-9]+p[0-9]+}}_c{{[0-9]+p[0-9]+}}_zicsr{{[0-9]+p[0-9]+}}
