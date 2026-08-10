// XFAIL: target={{.*}}-aix{{.*}}

// RUN: %clang -### -c -save-temps -integrated-as --target=x86_64 %s 2>&1 | FileCheck %s

// CHECK: cc1as
// CHECK-NOT: -mrelax-all

// RISC-V does not enable -mrelax-all
// RUN: %clang -### -c -save-temps -integrated-as --target=riscv64 %s 2>&1 | FileCheck %s -check-prefix=RISCV-RELAX

// RISCV-RELAX: cc1as
// RISCV-RELAX-NOT: -mrelax-all

// RUN: %clang -### -fintegrated-as -c -save-temps %s 2>&1 | FileCheck %s -check-prefix FIAS

// FIAS: cc1as

// RUN: %clang -target none -### -fno-integrated-as -S %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOFIAS

// NOFIAS-NOT: cc1as
// NOFIAS: -cc1
// NOFIAS: -no-integrated-as
