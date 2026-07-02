// Check that explicit -mcpu=<cpu>+no<feature> preserves -<feature> option and removes the feature when the feature is implied by the CPU or its base architecture. 
// SVE and RandGen are default features of neoverse-v2 (defined in AArch64Processors.td).
// SVE and SB are default features of armv9 (defined in AArch64Features.td).
// SHA2 is not part of the default feature set for either Neoverse V2 or Armv9.

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosve %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSVE
// NEOVERSE-V2-NOSVE: "-target-feature" "-sve"
// NEOVERSE-V2-NOSVE-NOT: "-target-feature" "+sve"
// NEOVERSE-V2-NOSVE: "-target-feature" "-sve2"
// NEOVERSE-V2-NOSVE-NOT: "-target-feature" "+sve"
// NEOVERSE-V2-NOSVE-NOT: "-target-feature" "+sve2"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+norng %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NORNG
// NEOVERSE-V2-NORNG: "-target-feature" "-rand"
// NEOVERSE-V2-NORNG-NOT: "-target-feature" "+rand"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosb %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSB
// NEOVERSE-V2-NOSB: "-target-feature" "-sb"
// NEOVERSE-V2-NOSB-NOT: "-target-feature" "+sb"

// RUN: %clang --target=aarch64-linux-gnu -mcpu=neoverse-v2+nosha2 %s -### 2>&1 | FileCheck %s --check-prefix=NEOVERSE-V2-NOSHA2
// NEOVERSE-V2-NOSHA2-NOT: "-target-feature" "+sha2"
// NEOVERSE-V2-NOSHA2-NOT: "-target-feature" "-sha2"
