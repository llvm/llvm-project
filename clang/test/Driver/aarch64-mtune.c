// Ensure we support the -mtune flag.

// GENERIC-CPU: "-target-cpu" "generic"

// There shouldn't be a default -mtune.
// RUN: %clang -### -c --target=aarch64 %s 2>&1 | FileCheck %s -check-prefix=NOTUNE
// NOTUNE-NOT: "-tune-cpu"

// RUN: %clang -### -c --target=aarch64 %s -mtune=generic 2>&1 | FileCheck %s -check-prefix=GENERIC
// GENERIC: "-tune-cpu" "generic"

// RUN: %clang -### -c --target=aarch64 %s -mtune=neoverse-n1 2>&1 | FileCheck %s --check-prefixes=GENERIC-CPU,NEOVERSE-N1
// NEOVERSE-N1: "-tune-cpu" "neoverse-n1"

// RUN: %clang -### -c --target=aarch64 %s -mtune=thunderx2t99 2>&1 | FileCheck %s -check-prefixes=GENERIC-CPU,THUNDERX2T99
// THUNDERX2T99: "-tune-cpu" "thunderx2t99"

// Check interaction between march and mtune.

// RUN: %clang -### -c --target=aarch64 %s -march=armv8-a 2>&1 | FileCheck %s --check-prefixes=GENERIC-CPU,MARCHARMV8A
// MARCHARMV8A-NOT: "-tune-cpu"

// RUN: %clang -### -c --target=aarch64 %s -march=armv8-a -mtune=cortex-a75 2>&1 | FileCheck %s --check-prefixes=GENERIC-CPU,MARCHARMV8A-A75
// MARCHARMV8A-A75: "-tune-cpu" "cortex-a75"

// Check interaction between mcpu and mtune.

// RUN: %clang -### -c --target=aarch64 %s -mcpu=thunderx 2>&1 | FileCheck %s -check-prefix=MCPUTHUNDERX
// MCPUTHUNDERX: "-target-cpu" "thunderx"
// MCPUTHUNDERX-NOT: "-tune-cpu"

// RUN: %clang -### -c --target=aarch64 %s -mcpu=cortex-a75 -mtune=cortex-a57 2>&1 | FileCheck %s -check-prefix=MCPUA75-MTUNEA57
// MCPUA75-MTUNEA57: "-target-cpu" "cortex-a75"
// MCPUA75-MTUNEA57: "-tune-cpu" "cortex-a57"
