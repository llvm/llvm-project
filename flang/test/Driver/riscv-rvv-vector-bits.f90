! -----------------------------------------------------------------------------
! Tests for the -mrvv-vector-bits flag (taken from the clang test)
! -----------------------------------------------------------------------------

! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-128 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=256 2>&1 | FileCheck --check-prefix=CHECK-256 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=512 2>&1 | FileCheck --check-prefix=CHECK-512 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=1024 2>&1 | FileCheck --check-prefix=CHECK-1024 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=2048 2>&1 | FileCheck --check-prefix=CHECK-2048 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-SCALABLE %s

! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv_zvl256b \
! RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-256 %s
! RUN: %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv_zvl512b \
! RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-512 %s

! CHECK-128: "-fc1"
! CHECK-128-SAME: "-mvscale-max=2" "-mvscale-min=2"
! CHECK-256: "-fc1"
! CHECK-256-SAME: "-mvscale-max=4" "-mvscale-min=4"
! CHECK-512: "-fc1"
! CHECK-512-SAME: "-mvscale-max=8" "-mvscale-min=8"
! CHECK-1024: "-fc1"
! CHECK-1024-SAME: "-mvscale-max=16" "-mvscale-min=16"
! CHECK-2048: "-fc1"
! CHECK-2048-SAME: "-mvscale-max=32" "-mvscale-min=32"

! CHECK-SCALABLE-NOT: "-mvscale-min=
! CHECK-SCALABLE-NOT: "-mvscale-max=

! Error out if an unsupported value is passed to -mrvv-vector-bits.
! -----------------------------------------------------------------------------
! RUN: not %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=16 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
! RUN: not %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=A 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
! RUN: not %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
! RUN:  -mrvv-vector-bits=131072 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
! RUN: not %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gc \
! RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
! RUN: not %flang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv \
! RUN:  -mrvv-vector-bits=64 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
!
! CHECK-BAD-VALUE-ERROR: error: unsupported argument '{{.*}}' to option '-mrvv-vector-bits='

