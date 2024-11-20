// RUN: %clang --target=riscv32-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-linux-android -### %s -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=ANDROID,DEFAULT,FAST-SCALAR-UNALIGNED-ACCESS,FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang -mabi=lp64d --target=riscv64-linux-android -### %s -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=ANDROID,DEFAULT,FAST-SCALAR-UNALIGNED-ACCESS,FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang -mabi=lp64d --target=riscv64-linux-android -mstrict-align -mvector-strict-align -### %s -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=NO-FAST-SCALAR-UNALIGNED-ACCESS,NO-FAST-VECTOR-UNALIGNED-ACCESS


// CHECK: fno-signed-char

// RUN: %clang --target=riscv32-unknown-elf -### %s 2>&1 | FileCheck %s -check-prefix=DEFAULT

// RUN: %clang --target=riscv32-unknown-elf -### %s -mrelax 2>&1 | FileCheck %s -check-prefix=RELAX
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-relax 2>&1 | FileCheck %s -check-prefix=NO-RELAX

// ANDROID: "-target-feature" "+v"
// ANDROID: "-target-feature" "+zba"
// ANDROID: "-target-feature" "+zbb"
// ANDROID: "-target-feature" "+zbs"
// RELAX: "-target-feature" "+relax"
// NO-RELAX: "-target-feature" "-relax"
// DEFAULT: "-target-feature" "+relax"
// DEFAULT-NOT: "-target-feature" "-relax"

// RUN: %clang --target=riscv32-unknown-elf -### %s -msave-restore 2>&1 | FileCheck %s -check-prefix=SAVE-RESTORE
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-save-restore 2>&1 | FileCheck %s -check-prefix=NO-SAVE-RESTORE

// SAVE-RESTORE: "-target-feature" "+save-restore"
// NO-SAVE-RESTORE: "-target-feature" "-save-restore"
// DEFAULT-NOT: "-target-feature" "-save-restore"
// DEFAULT-NOT: "-target-feature" "+save-restore"

// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-strict-align 2>&1 | FileCheck %s -check-prefixes=FAST-SCALAR-UNALIGNED-ACCESS,FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mstrict-align 2>&1 | FileCheck %s -check-prefixes=NO-FAST-SCALAR-UNALIGNED-ACCESS,NO-FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-scalar-strict-align 2>&1 | FileCheck %s -check-prefix=FAST-SCALAR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mscalar-strict-align 2>&1 | FileCheck %s -check-prefix=NO-FAST-SCALAR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-scalar-strict-align -mstrict-align 2>&1 | FileCheck %s -check-prefixes=NO-FAST-SCALAR-UNALIGNED-ACCESS,NO-FAST-VECTOR-UNALIGNED-ACCESS
// RUN: touch %t.o
// RUN: %clang --target=riscv32-unknown-elf -### %t.o -mno-strict-align -mstrict-align

// FAST-SCALAR-UNALIGNED-ACCESS: "-target-feature" "+unaligned-scalar-mem"
// NO-FAST-SCALAR-UNALIGNED-ACCESS: "-target-feature" "-unaligned-scalar-mem"

// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-vector-strict-align 2>&1 | FileCheck %s -check-prefix=FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mvector-strict-align 2>&1 | FileCheck %s -check-prefix=NO-FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-vector-strict-align -mstrict-align 2>&1 | FileCheck %s -check-prefix=NO-FAST-VECTOR-UNALIGNED-ACCESS
// RUN: %clang --target=riscv32-unknown-elf -### %s -mno-strict-align -mvector-strict-align 2>&1 | FileCheck %s -check-prefix=NO-FAST-VECTOR-UNALIGNED-ACCESS
// FAST-VECTOR-UNALIGNED-ACCESS: "-target-feature" "+unaligned-vector-mem"
// NO-FAST-VECTOR-UNALIGNED-ACCESS: "-target-feature" "-unaligned-vector-mem"

// RUN: %clang --target=riscv32-unknown-elf -### %s 2>&1 | FileCheck %s -check-prefix=NOUWTABLE
// RUN: %clang --target=riscv32-unknown-elf -fasynchronous-unwind-tables -### %s 2>&1 | FileCheck %s -check-prefix=UWTABLE
// RUN: %clang --target=riscv64-unknown-elf -### %s 2>&1 | FileCheck %s -check-prefix=NOUWTABLE
// RUN: %clang --target=riscv64-unknown-elf -fasynchronous-unwind-tables -### %s 2>&1 | FileCheck %s -check-prefix=UWTABLE
//
// UWTABLE: "-funwind-tables=2"
// NOUWTABLE-NOT: "-funwind-tables=2"

// RUN: %clang --target=riscv32-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DEFAULT-LINUX
// RUN: %clang --target=riscv64-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DEFAULT-LINUX

// DEFAULT-LINUX: "-funwind-tables=2"
// DEFAULT-LINUX-SAME: "-target-feature" "+m"
// DEFAULT-LINUX-SAME: "-target-feature" "+a"
// DEFAULT-LINUX-SAME: "-target-feature" "+f"
// DEFAULT-LINUX-SAME: "-target-feature" "+d"
// DEFAULT-LINUX-SAME: "-target-feature" "+c"

// RUN: not %clang -c --target=riscv64-linux-gnu -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: not %clang -c --target=riscv64 -gsplit-dwarf=single %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: %clang -### -c --target=riscv64 -mno-relax -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=SPLIT-DWARF

// ERR-SPLIT-DWARF: error: -gsplit-dwarf{{.*}} is unsupported with RISC-V linker relaxation (-mrelax)
// SPLIT-DWARF:     "-split-dwarf-file"
