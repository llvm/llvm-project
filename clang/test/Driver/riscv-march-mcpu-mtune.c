// Check the priority between -mcpu, -mtune and -march

// sifive-e76 is rv32imafc and sifive-e31 is rv32imac

// -mcpu, -mtune and -march are not given, pipeline model and arch ext. use
// default setting.
// RUN: %clang --target=riscv32-elf -### -c %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT: "-target-cpu" "generic-rv32"
// CHECK-DEFAULT: "-target-feature" "+m" "-target-feature" "+a"
// CHECK-DEFAULT: "-target-feature" "+c"

// -mtune is given, pipeline model take from -mtune, arch ext. use
// default setting.
// RUN: %clang --target=riscv32 -mtune=sifive-e76 -### -c %s 2>&1 \
// RUN:     | FileCheck -check-prefix=MTUNE-E76 %s
// MTUNE-E76: "-target-feature" "+m" "-target-feature" "+a"
// MTUNE-E76: "-target-feature" "+c"
// MTUNE-E76: "-target-feature" "-f"
// MTUNE-E76: "-tune-cpu" "sifive-e76"

// -march is given, arch ext. take from -march, pipeline model use
// default setting.
// RUN: %clang --target=riscv32 -### -c %s -march=rv32imafdc 2>&1 \
// RUN:     | FileCheck -check-prefix=MARCH-RV32IMAFDC %s
// MARCH-RV32IMAFDC: "-target-cpu" "generic-rv32"
// MARCH-RV32IMAFDC: "-target-feature" "+m" "-target-feature" "+a"
// MARCH-RV32IMAFDC: "-target-feature" "+f" "-target-feature" "+d"
// MARCH-RV32IMAFDC: "-target-feature" "+c"

// -mcpu is given, pipeline model and arch ext. from -mcpu.
// RUN: %clang --target=riscv32 -### -c %s -mcpu=sifive-e76 2>&1 \
// RUN:     | FileCheck -check-prefix=MCPU-E76 %s
// MCPU-E76: "-target-cpu" "sifive-e76"
// MCPU-E76: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-E76: "-target-feature" "+f" "-target-feature" "+c"

// -mcpu and -mtune are given, so pipeline model take from -mtune, and arch ext.
// take from -mcpu since -march is not given.
// RUN: %clang --target=riscv32 -### -c %s -mcpu=sifive-e76 -mtune=sifive-e31 2>&1 \
// RUN:     | FileCheck -check-prefix=MCPU-E76-MTUNE-E31 %s
// MCPU-E76-MTUNE-E31: "-target-cpu" "sifive-e76"
// MCPU-E76-MTUNE-E31: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-E76-MTUNE-E31: "-target-feature" "+f" "-target-feature" "+c"
// MCPU-E76-MTUNE-E31: "-tune-cpu" "sifive-e31"

// RUN: %clang --target=riscv32 -### -c %s -mtune=sifive-e76 -mcpu=sifive-e31 2>&1 \
// RUN:     | FileCheck -check-prefix=MTUNE-E76-MCPU-E31 %s
// MTUNE-E76-MCPU-E31: "-target-cpu" "sifive-e31"
// MTUNE-E76-MCPU-E31: "-target-feature" "+m" "-target-feature" "+a"
// MTUNE-E76-MCPU-E31: "-target-feature" "+c"
// MTUNE-E76-MCPU-E31: "-target-feature" "-f"
// MTUNE-E76-MCPU-E31: "-tune-cpu" "sifive-e76"

// -mcpu and -march are given, so pipeline model take from -mcpu since -mtune is
// not given, and arch ext. take from -march.
// RUN: %clang --target=riscv32 -### -c %s -mcpu=sifive-e31 -march=rv32ic 2>&1 \
// RUN:     | FileCheck -check-prefix=MCPU-E31-MARCH-RV32I %s
// MCPU-E31-MARCH-RV32I: "-target-cpu" "sifive-e31"
// MCPU-E31-MARCH-RV32I: "-target-feature" "+c"
// MCPU-E31-MARCH-RV32I: "-target-feature" "-a"
// MCPU-E31-MARCH-RV32I: "-target-feature" "-f"
// MCPU-E31-MARCH-RV32I: "-target-feature" "-m"

// -mcpu, -march and -mtune are given, so pipeline model take from -mtune
// and arch ext. take from -march, -mcpu is unused.
// RUN: %clang --target=riscv32 -### -c %s -mcpu=sifive-e31 -mtune=sifive-e76 -march=rv32ic 2>&1 \
// RUN:     | FileCheck -check-prefix=MCPU-E31-MTUNE-E76-MARCH-RV32I %s
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-target-cpu" "sifive-e31"
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-target-feature" "+c"
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-target-feature" "-a"
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-target-feature" "-f"
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-target-feature" "-m"
// MCPU-E31-MTUNE-E76-MARCH-RV32I: "-tune-cpu" "sifive-e76"
