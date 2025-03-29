// RUN: %clang -### --target=aarch64-none-elf -march=armv8.4a+memtag %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.5a+memtag %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=aarch64-none-elf -mcpu=cortex-a510 %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+mte"

// RUN: %clang -### --target=aarch64-none-elf -mcpu=cortex-a510+nomemtag %s 2>&1 | FileCheck %s --check-prefix=NOMTE
// NOMTE: "-target-feature" "-mte"

// RUN: %clang -### --target=aarch64-none-elf                 %s 2>&1 | FileCheck %s --check-prefix=ABSENTMTE
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.4a %s 2>&1 | FileCheck %s --check-prefix=ABSENTMTE
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.5a %s 2>&1 | FileCheck %s --check-prefix=ABSENTMTE
// ABSENTMTE-NOT: "-target-feature" "+mte"
// ABSENTMTE-NOT: "-target-feature" "-mte"
