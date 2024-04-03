// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve %s -### 2>&1 | FileCheck %s --check-prefix=SVE-ONLY
// SVE-ONLY: "-target-feature" "+sve"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+nosve %s -### 2>&1 | FileCheck %s --check-prefix=NOSVE
// NOSVE-NOT: sve"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve+nosve %s -### 2>&1 | FileCheck %s --check-prefix=SVE-REVERT
// SVE-REVERT-NOT: "-target-feature" "+sve"
// SVE-REVERT: "-target-feature" "-sve"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-IMPLY
// SVE2-IMPLY: "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2+nosve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-REVERT
// SVE2-REVERT: "-target-feature" "+sve"{{.*}} "-target-feature" "-sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2+nosve %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-CONFLICT
// SVE2-CONFLICT: "-target-feature" "-sve" "-target-feature" "-sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+nosve+sve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-CONFLICT-REV
// SVE2-CONFLICT-REV: "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve+sve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE-SVE2
// SVE-SVE2: "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-bitperm %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-BITPERM
// SVE2-BITPERM: "-target-feature" "+sve" "-target-feature" "+sve2-bitperm" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+nosve2-bitperm %s -### 2>&1 | FileCheck %s --check-prefix=NOSVE2-BITPERM
// NOSVE2-BITPERM-NOT: "-target-feature" "+sve2-bitperm"
// NOSVE2-BITPERM-NOT: "-target-feature" "+sve2"
// NOSVE2-BITPERM-NOT: "-target-feature" "+sve"
// NOSVE2-BITPERM-NOT: sve2-bitperm"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-bitperm+nosve2-bitperm %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-BITPERM-REVERT
// SVE2-BITPERM-REVERT: "-target-feature" "+sve" "-target-feature" "-sve2-bitperm" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-aes+nosve2-aes %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-AES-REVERT
// SVE2-AES-REVERT: "-target-feature" "+sve" "-target-feature" "-sve2-aes" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-sha3+nosve2-sha3 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SHA3-REVERT
// SVE2-SHA3-REVERT: "-target-feature" "+sve" "-target-feature" "-sve2-sha3" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-sm4+nosve2-sm4 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SM4-REVERT
// SVE2-SM4-REVERT: "-target-feature" "+sve" "-target-feature" "-sve2-sm4" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-sha3 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SHA3
// SVE2-SHA3: "-target-feature" "+sve" "-target-feature" "+sve2-sha3" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-aes %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-AES
// SVE2-AES: "-target-feature" "+sve" "-target-feature" "+sve2-aes" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-sm4 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SM4
// SVE2-SM4: "-target-feature" "+sve" "-target-feature" "+sve2-sm4" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-bitperm+nosve2-aes %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SUBFEATURE-MIX
// SVE2-SUBFEATURE-MIX: "-target-feature" "+sve" "-target-feature" "+sve2-bitperm" "-target-feature" "+sve2"
// SVE2-SUBFEATURE-NOT: sve2-aes

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-sm4+nosve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE2-SUBFEATURE-CONFLICT
// SVE2-SUBFEATURE-CONFLICT: "-target-feature" "+sve" "-target-feature" "-sve2-sm4" "-target-feature" "-sve2"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-aes+nosve %s -### 2>&1 | FileCheck %s --check-prefix=SVE-SUBFEATURE-CONFLICT
// SVE-SUBFEATURE-CONFLICT-NOT: "-target-feature" "+sve2-aes"
// SVE-SUBFEATURE-CONFLICT-NOT: "-target-feature" "+sve2"
// SVE-SUBFEATURE-CONFLICT-NOT: "-target-feature" "+sve"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+nosve+sve2-aes %s -### 2>&1 | FileCheck %s --check-prefix=SVE-SUBFEATURE-CONFLICT-REV
// SVE-SUBFEATURE-CONFLICT-REV: "-target-feature" "+sve" "-target-feature" "+sve2-aes" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-n2+nosve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE-MCPU-FEATURES
// SVE-MCPU-FEATURES-NOT: "-target-feature" "+sve2-bitperm"
// SVE-MCPU-FEATURES-NOT: "-target-feature" "+sve2"
// SVE-MCPU-FEATURES: "-target-feature" "+sve"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-n2+nosve+sve2 %s -### 2>&1 | FileCheck %s --check-prefix=SVE-MCPU-FEATURES-CONFLICT
// SVE-MCPU-FEATURES-CONFLICT-NOT: "-target-feature" "+sve2-bitperm"
// SVE-MCPU-FEATURES-CONFLICT: "-target-feature" "+sve"
// SVE-MCPU-FEATURES-CONFLICT-SAME: "-target-feature" "+sve2"
