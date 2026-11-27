// Test that -mcpu=help and -mtune=help list different names for RISC-V.

// REQUIRES: riscv-registered-target

// RUN: %clang --target=riscv32-unknown-linux-gnu --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV32-CPU
// RUN: %clang --target=riscv32-unknown-linux-gnu -mcpu=help 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV32-CPU

// RUN: %clang --target=riscv32-unknown-linux-gnu --print-supported-tune-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV32-TUNE
// RUN: %clang --target=riscv32-unknown-linux-gnu -mtune=help 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV32-TUNE

// RUN: %clang --target=riscv64-unknown-linux-gnu --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV64-CPU
// RUN: %clang --target=riscv64-unknown-linux-gnu -mcpu=help 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV64-CPU

// RUN: %clang --target=riscv64-unknown-linux-gnu --print-supported-tune-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV64-TUNE
// RUN: %clang --target=riscv64-unknown-linux-gnu -mtune=help 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-RV64-TUNE

// CHECK-RV32-CPU: Target: riscv32-unknown-linux-gnu
// CHECK-RV32-CPU: Available CPUs for this target:
// CHECK-RV32-CPU-EMPTY:
// CHECK-RV32-CPU-NEXT: andes-a25
// CHECK-RV32-CPU-NEXT: andes-a45
// CHECK-RV32-CPU-NEXT: andes-n45
// CHECK-RV32-CPU-NEXT: chipsalliance-veer-eh1
// CHECK-RV32-CPU-NEXT: generic-rv32
// CHECK-RV32-CPU-NEXT: rocket-rv32
// CHECK-RV32-CPU-NEXT: rp2350-hazard3
// CHECK-RV32-CPU-NEXT: sifive-e20
// CHECK-RV32-CPU-NEXT: sifive-e21
// CHECK-RV32-CPU-NEXT: sifive-e24
// CHECK-RV32-CPU-NEXT: sifive-e31
// CHECK-RV32-CPU-NEXT: sifive-e34
// CHECK-RV32-CPU-NEXT: sifive-e76
// CHECK-RV32-CPU-NEXT: sifive-x160
// CHECK-RV32-CPU-NEXT: syntacore-scr1-base
// CHECK-RV32-CPU-NEXT: syntacore-scr1-max
// CHECK-RV32-CPU-NEXT: syntacore-scr3-rv32
// CHECK-RV32-CPU-NEXT: syntacore-scr4-rv32
// CHECK-RV32-CPU-NEXT: syntacore-scr5-rv32
// CHECK-RV32-CPU-EMPTY:
// CHECK-RV32-CPU-NEXT: Use -mcpu or -mtune to specify the target's processor.

// CHECK-RV32-TUNE: Target: riscv32-unknown-linux-gnu
// CHECK-RV32-TUNE: Available tuning CPUs for this target:
// CHECK-RV32-TUNE-EMPTY:
// CHECK-RV32-TUNE-NEXT: andes-45-series
// CHECK-RV32-TUNE-NEXT: andes-a25
// CHECK-RV32-TUNE-NEXT: andes-a45
// CHECK-RV32-TUNE-NEXT: andes-n45
// CHECK-RV32-TUNE-NEXT: chipsalliance-veer-eh1
// CHECK-RV32-TUNE-NEXT: generic
// CHECK-RV32-TUNE-NEXT: generic-ooo
// CHECK-RV32-TUNE-NEXT: generic-rv32
// CHECK-RV32-TUNE-NEXT: rocket
// CHECK-RV32-TUNE-NEXT: rocket-rv32
// CHECK-RV32-TUNE-NEXT: rp2350-hazard3
// CHECK-RV32-TUNE-NEXT: sifive-7-series
// CHECK-RV32-TUNE-NEXT: sifive-e20
// CHECK-RV32-TUNE-NEXT: sifive-e21
// CHECK-RV32-TUNE-NEXT: sifive-e24
// CHECK-RV32-TUNE-NEXT: sifive-e31
// CHECK-RV32-TUNE-NEXT: sifive-e34
// CHECK-RV32-TUNE-NEXT: sifive-e76
// CHECK-RV32-TUNE-NEXT: sifive-x160
// CHECK-RV32-TUNE-NEXT: syntacore-scr1-base
// CHECK-RV32-TUNE-NEXT: syntacore-scr1-max
// CHECK-RV32-TUNE-NEXT: syntacore-scr3-rv32
// CHECK-RV32-TUNE-NEXT: syntacore-scr4-rv32
// CHECK-RV32-TUNE-NEXT: syntacore-scr5-rv32
// CHECK-RV32-TUNE-EMPTY:
// CHECK-RV32-TUNE-NEXT: Use -mtune to specify the target's tuning processor.

// CHECK-RV64-CPU: Target: riscv64-unknown-linux-gnu
// CHECK-RV64-CPU: Available CPUs for this target:
// CHECK-RV64-CPU-EMPTY:
// CHECK-RV64-CPU-NEXT: an-erbium
// CHECK-RV64-CPU-NEXT: andes-ax25
// CHECK-RV64-CPU-NEXT: andes-ax45
// CHECK-RV64-CPU-NEXT: andes-ax45mpv
// CHECK-RV64-CPU-NEXT: andes-nx45
// CHECK-RV64-CPU-NEXT: et-soc1
// CHECK-RV64-CPU-NEXT: gaisler-gr765
// CHECK-RV64-CPU-NEXT: generic-rv64
// CHECK-RV64-CPU-NEXT: mips-p8700
// CHECK-RV64-CPU-NEXT: rocket-rv64
// CHECK-RV64-CPU-NEXT: sifive-p450
// CHECK-RV64-CPU-NEXT: sifive-p470
// CHECK-RV64-CPU-NEXT: sifive-p550
// CHECK-RV64-CPU-NEXT: sifive-p670
// CHECK-RV64-CPU-NEXT: sifive-p870-d
// CHECK-RV64-CPU-NEXT: sifive-s21
// CHECK-RV64-CPU-NEXT: sifive-s51
// CHECK-RV64-CPU-NEXT: sifive-s54
// CHECK-RV64-CPU-NEXT: sifive-s76
// CHECK-RV64-CPU-NEXT: sifive-u54
// CHECK-RV64-CPU-NEXT: sifive-u74
// CHECK-RV64-CPU-NEXT: sifive-x180
// CHECK-RV64-CPU-NEXT: sifive-x280
// CHECK-RV64-CPU-NEXT: sifive-x390
// CHECK-RV64-CPU-NEXT: spacemit-a100
// CHECK-RV64-CPU-NEXT: spacemit-x60
// CHECK-RV64-CPU-NEXT: spacemit-x100
// CHECK-RV64-CPU-NEXT: syntacore-scr3-rv64
// CHECK-RV64-CPU-NEXT: syntacore-scr4-rv64
// CHECK-RV64-CPU-NEXT: syntacore-scr5-rv64
// CHECK-RV64-CPU-NEXT: syntacore-scr7
// CHECK-RV64-CPU-NEXT: tt-ascalon-x
// CHECK-RV64-CPU-NEXT: tt-ascalon-xg
// CHECK-RV64-CPU-NEXT: xiangshan-kunminghu
// CHECK-RV64-CPU-NEXT: xiangshan-nanhu
// CHECK-RV64-CPU-NEXT: xt-c910v2
// CHECK-RV64-CPU-NEXT: xt-c920v2
// CHECK-RV64-CPU-EMPTY:
// CHECK-RV64-CPU-NEXT: Use -mcpu or -mtune to specify the target's processor.

// CHECK-RV64-TUNE: Target: riscv64-unknown-linux-gnu
// CHECK-RV64-TUNE: Available tuning CPUs for this target:
// CHECK-RV64-TUNE-EMPTY:
// CHECK-RV64-TUNE-NEXT: an-erbium
// CHECK-RV64-TUNE-NEXT: andes-45-series
// CHECK-RV64-TUNE-NEXT: andes-ax25
// CHECK-RV64-TUNE-NEXT: andes-ax45
// CHECK-RV64-TUNE-NEXT: andes-ax45mpv
// CHECK-RV64-TUNE-NEXT: andes-nx45
// CHECK-RV64-TUNE-NEXT: et-soc1
// CHECK-RV64-TUNE-NEXT: gaisler-gr765
// CHECK-RV64-TUNE-NEXT: generic
// CHECK-RV64-TUNE-NEXT: generic-ooo
// CHECK-RV64-TUNE-NEXT: generic-rv64
// CHECK-RV64-TUNE-NEXT: mips-p8700
// CHECK-RV64-TUNE-NEXT: rocket
// CHECK-RV64-TUNE-NEXT: rocket-rv64
// CHECK-RV64-TUNE-NEXT: sifive-7-series
// CHECK-RV64-TUNE-NEXT: sifive-p450
// CHECK-RV64-TUNE-NEXT: sifive-p470
// CHECK-RV64-TUNE-NEXT: sifive-p550
// CHECK-RV64-TUNE-NEXT: sifive-p670
// CHECK-RV64-TUNE-NEXT: sifive-p870-d
// CHECK-RV64-TUNE-NEXT: sifive-s21
// CHECK-RV64-TUNE-NEXT: sifive-s51
// CHECK-RV64-TUNE-NEXT: sifive-s54
// CHECK-RV64-TUNE-NEXT: sifive-s76
// CHECK-RV64-TUNE-NEXT: sifive-u54
// CHECK-RV64-TUNE-NEXT: sifive-u74
// CHECK-RV64-TUNE-NEXT: sifive-x180
// CHECK-RV64-TUNE-NEXT: sifive-x280
// CHECK-RV64-TUNE-NEXT: sifive-x390
// CHECK-RV64-TUNE-NEXT: spacemit-a100
// CHECK-RV64-TUNE-NEXT: spacemit-x60
// CHECK-RV64-TUNE-NEXT: spacemit-x100
// CHECK-RV64-TUNE-NEXT: syntacore-scr3-rv64
// CHECK-RV64-TUNE-NEXT: syntacore-scr4-rv64
// CHECK-RV64-TUNE-NEXT: syntacore-scr5-rv64
// CHECK-RV64-TUNE-NEXT: syntacore-scr7
// CHECK-RV64-TUNE-NEXT: tt-ascalon-x
// CHECK-RV64-TUNE-NEXT: tt-ascalon-xg
// CHECK-RV64-TUNE-NEXT: xiangshan-kunminghu
// CHECK-RV64-TUNE-NEXT: xiangshan-nanhu
// CHECK-RV64-TUNE-NEXT: xt-c910v2
// CHECK-RV64-TUNE-NEXT: xt-c920v2
// CHECK-RV64-TUNE-EMPTY:
// CHECK-RV64-TUNE-NEXT: Use -mtune to specify the target's tuning processor.
