// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s

// CHECK-NOT: __riscv_xtheadba {{.*$}}
// CHECK-NOT: __riscv_xtheadbb {{.*$}}
// CHECK-NOT: __riscv_xtheadbs {{.*$}}
// CHECK-NOT: __riscv_xtheadcmo {{.*$}}
// CHECK-NOT: __riscv_xtheadcondmov {{.*$}}
// CHECK-NOT: __riscv_xtheadfmemidx {{.*$}}
// CHECK-NOT: __riscv_xtheadmac {{.*$}}
// CHECK-NOT: __riscv_xtheadmemidx {{.*$}}
// CHECK-NOT: __riscv_xtheadmempair {{.*$}}
// CHECK-NOT: __riscv_xtheadsync {{.*$}}
// CHECK-NOT: __riscv_xtheadvdot {{.*$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadba -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadba -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBA-EXT %s
// CHECK-XTHEADBA-EXT: __riscv_xtheadba 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadbb -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadbb -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBB-EXT %s
// CHECK-XTHEADBB-EXT: __riscv_xtheadbb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadbs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadbs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADBS-EXT %s
// CHECK-XTHEADBS-EXT: __riscv_xtheadbs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadcmo -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADCMO-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadcmo -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADCMO-EXT %s
// CHECK-XTHEADCMO-EXT: __riscv_xtheadcmo 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadcondmov -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADCONDMOV-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadcondmov -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADCONDMOV-EXT %s
// CHECK-XTHEADCONDMOV-EXT: __riscv_xtheadcondmov 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadfmemidx -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADFMEMIDX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadfmemidx -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADFMEMIDX-EXT %s
// CHECK-XTHEADFMEMIDX-EXT: __riscv_xtheadfmemidx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadmac -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMAC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadmac -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMAC-EXT %s
// CHECK-XTHEADMAC-EXT: __riscv_xtheadmac 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadmemidx -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMEMIDX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadmemidx -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMEMIDX-EXT %s
// CHECK-XTHEADMEMIDX-EXT: __riscv_xtheadmemidx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadmempair -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMEMPAIR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadmempair -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADMEMPAIR-EXT %s
// CHECK-XTHEADMEMPAIR-EXT: __riscv_xtheadmempair 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadsync -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADSYNC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadsync -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADSYNC-EXT %s
// CHECK-XTHEADSYNC-EXT: __riscv_xtheadsync 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixtheadvdot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADVDOT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixtheadvdot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XTHEADVDOT-EXT %s
// CHECK-XTHEADVDOT-EXT: __riscv_xtheadvdot 1000000{{$}}
