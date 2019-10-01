// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-MEDLOW %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-MEDLOW %s

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -mcmodel=small -o - | FileCheck --check-prefix=CHECK-MEDLOW %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -mcmodel=small -o - | FileCheck --check-prefix=CHECK-MEDLOW %s

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -mcmodel=medlow -o - | FileCheck --check-prefix=CHECK-MEDLOW %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -mcmodel=medlow -o - | FileCheck --check-prefix=CHECK-MEDLOW %s

// CHECK-MEDLOW: #define __riscv_cmodel_medlow 1
// CHECK-MEDLOW-NOT: __riscv_cmodel_medany

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -mcmodel=medium -o - | FileCheck --check-prefix=CHECK-MEDANY %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -mcmodel=medium -o - | FileCheck --check-prefix=CHECK-MEDANY %s

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -mcmodel=medany -o - | FileCheck --check-prefix=CHECK-MEDANY %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -mcmodel=medany -o - | FileCheck --check-prefix=CHECK-MEDANY %s

// CHECK-MEDANY: #define __riscv_cmodel_medany 1
// CHECK-MEDANY-NOT: __riscv_cmodel_medlow
