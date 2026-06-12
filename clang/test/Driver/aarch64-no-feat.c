// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2 %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-DEFAULT
// NEOVERSE-V2-DEFAULT-NOT: "-target-feature" "+sb"
// NEOVERSE-V2-DEFAULT-NOT: "-target-feature" "-sb"
// NEOVERSE-V2-DEFAULT: "-target-feature" "+rand"
// NEOVERSE-V2-DEFAULT-SAME: "-target-feature" "+sve"
// NEOVERSE-V2-DEFAULT-NOT: "-target-feature" "+sha2"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosb %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSB
// NEOVERSE-V2-NOSB: "-target-feature" "-sb"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+sb %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-SB
// NEOVERSE-V2-SB-NOT: "-target-feature" "+sb"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosve %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSVE
// NEOVERSE-V2-NOSVE: "-target-feature" "-sve"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+sve %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-SVE
// NEOVERSE-V2-SVE: "-target-feature" "+sve"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+norng %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NORNG
// NEOVERSE-V2-NORNG: "-target-feature" "-rand"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+rng %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-RNG
// NEOVERSE-V2-RNG: "-target-feature" "+rand"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosha2 %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSHA2
// NEOVERSE-V2-NOSHA2-NOT: "-target-feature" "+sha2"
// NEOVERSE-V2-NOSHA2-NOT: "-target-feature" "-sha2"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+sha2 %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-SHA2
// NEOVERSE-V2-SHA2: "-target-feature" "+sha2"
