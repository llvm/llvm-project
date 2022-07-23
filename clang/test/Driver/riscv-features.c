// RUN: %clang --target=riscv32-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: fno-signed-char

// RUN: %clang --target=riscv32-unknown-elf -### %s 2>&1 | FileCheck %s -check-prefix=DEFAULT

// RUN: %clang --target=riscv32-unknown-elf -### %s -mrelax 2>&1 | FileCheck %s -check-prefix=RELAX
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-relax 2>&1 | FileCheck %s -check-prefix=NO-RELAX

// RELAX: "-target-feature" "+relax"
// NO-RELAX: "-target-feature" "-relax"
// DEFAULT: "-target-feature" "+relax"
// DEFAULT-NOT: "-target-feature" "-relax"

// RUN: %clang --target=riscv32-unknown-elf -### %s -msave-restore 2>&1 | FileCheck %s -check-prefix=SAVE-RESTORE
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-save-restore 2>&1 | FileCheck %s -check-prefix=NO-SAVE-RESTORE

// SAVE-RESTORE: "-target-feature" "+save-restore"
// NO-SAVE-RESTORE: "-target-feature" "-save-restore"
// DEFAULT: "-target-feature" "-save-restore"
// DEFAULT-NOT: "-target-feature" "+save-restore"

// RUN: %clang --target=riscv32-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DEFAULT-LINUX
// RUN: %clang --target=riscv64-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DEFAULT-LINUX

// DEFAULT-LINUX: "-target-feature" "+m"
// DEFAULT-LINUX-SAME: "-target-feature" "+a"
// DEFAULT-LINUX-SAME: "-target-feature" "+f"
// DEFAULT-LINUX-SAME: "-target-feature" "+d"
// DEFAULT-LINUX-SAME: "-target-feature" "+c"

// RUN: not %clang -cc1 -triple riscv64-unknown-elf -target-feature +e 2>&1 | FileCheck %s -check-prefix=RV64-WITH-E

// RV64-WITH-E: error: invalid feature combination: standard user-level extension 'e' requires 'rv32'

// RUN: not %clang -c --target=riscv64-linux-gnu -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: not %clang -c --target=riscv64 -gsplit-dwarf=single %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: %clang -### -c --target=riscv64 -mno-relax -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=SPLIT-DWARF

// ERR-SPLIT-DWARF: error: -gsplit-dwarf{{.*}} is unsupported with RISC-V linker relaxation (-mrelax)
// SPLIT-DWARF:     "-split-dwarf-file"
