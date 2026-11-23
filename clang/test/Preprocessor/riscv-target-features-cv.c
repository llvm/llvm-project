// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s

// CHECK-NOT: __riscv_xcvalu {{.*$}}
// CHECK-NOT: __riscv_xcvbi {{.*$}}
// CHECK-NOT: __riscv_xcvbitmanip {{.*$}}
// CHECK-NOT: __riscv_xcvelw {{.*$}}
// CHECK-NOT: __riscv_xcvmac {{.*$}}
// CHECK-NOT: __riscv_xcvmem {{.*$}}
// CHECK-NOT: __riscv_xcvsimd {{.*$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvalu -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVALU-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvalu -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVALU-EXT %s
// CHECK-XCVALU-EXT: __riscv_xcvalu 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvbi -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVBI-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvbi -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVBI-EXT %s
// CHECK-XCVBI-EXT: __riscv_xcvbi 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvbitmanip -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVBITMANIP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvbitmanip -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVBITMANIP-EXT %s
// CHECK-XCVBITMANIP-EXT: __riscv_xcvbitmanip 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvmac -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVMAC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvmac -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVMAC-EXT %s
// CHECK-XCVMAC-EXT: __riscv_xcvmac 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvmem -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVMEM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvmem -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVMEM-EXT %s
// CHECK-XCVMEM-EXT: __riscv_xcvmem 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixcvsimd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVSIMD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixcvsimd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XCVSIMD-EXT %s
// CHECK-XCVSIMD-EXT: __riscv_xcvsimd 1000000{{$}}
