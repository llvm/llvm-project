// Ensure we support the -mtune flag.

// There shouldn't be a default -mtune.
// RUN: %clang -target aarch64-unknown-unknown -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NOTUNE
// NOTUNE-NOT: "-tune-cpu" "generic"

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -mtune=generic 2>&1 \
// RUN:   | FileCheck %s -check-prefix=GENERIC
// GENERIC: "-tune-cpu" "generic"

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -mtune=neoverse-n1 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NEOVERSE-N1
// NEOVERSE-N1: "-tune-cpu" "neoverse-n1"

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -mtune=thunderx2t99 2>&1 \
// RUN:   | FileCheck %s -check-prefix=THUNDERX2T99
// THUNDERX2T99: "-tune-cpu" "thunderx2t99"

// Check interaction between march and mtune.

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -march=armv8-a 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MARCHARMV8A
// MARCHARMV8A: "-target-cpu" "generic"
// MARCHARMV8A-NOT: "-tune-cpu" "generic"

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -march=armv8-a -mtune=cortex-a75 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MARCHARMV8A-A75
// MARCHARMV8A-A75: "-target-cpu" "generic"
// MARCHARMV8A-A75: "-tune-cpu" "cortex-a75"

// Check interaction between mcpu and mtune.

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -mcpu=thunderx 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MCPUTHUNDERX
// MCPUTHUNDERX: "-target-cpu" "thunderx"
// MCPUTHUNDERX-NOT: "-tune-cpu"

// RUN: %clang -target aarch64-unknown-unknown -c -### %s -mcpu=cortex-a75 -mtune=cortex-a57 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MCPUA75-MTUNEA57
// MCPUA75-MTUNEA57: "-target-cpu" "cortex-a75"
// MCPUA75-MTUNEA57: "-tune-cpu" "cortex-a57"
