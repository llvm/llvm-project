// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s

// CHECK-NOT: __riscv_xandesperf {{.*$}}
// CHECK-NOT: __riscv_xandesvbfhcvt {{.*$}}
// CHECK-NOT: __riscv_xandesvpackfph {{.*$}}
// CHECK-NOT: __riscv_xandesvdot {{.*$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_xandesperf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESPERF %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_xandesperf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESPERF %s
// CHECK-XANDESPERF: __riscv_xandesperf  5000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_xandesvbfhcvt -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVBFHCVT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_xandesvbfhcvt -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVBFHCVT %s
// CHECK-XANDESVBFHCVT: __riscv_xandesvbfhcvt  5000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_xandesvpackfph -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVPACKFPH %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_xandesvpackfph -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVPACKFPH %s
// CHECK-XANDESVPACKFPH: __riscv_xandesvpackfph  5000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_xandesvdot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVDOT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_xandesvdot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XANDESVDOT %s
// CHECK-XANDESVDOT: __riscv_xandesvdot  5000000{{$}}
