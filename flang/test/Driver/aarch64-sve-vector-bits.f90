! -----------------------------------------------------------------------------
! Tests for the -msve-vector-bits flag (taken from the clang test)
! -----------------------------------------------------------------------------

! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-128 %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=256 2>&1 | FileCheck --check-prefix=CHECK-256 %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=512 2>&1 | FileCheck --check-prefix=CHECK-512 %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=1024 2>&1 | FileCheck --check-prefix=CHECK-1024 %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=2048 2>&1 | FileCheck --check-prefix=CHECK-2048 %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=128+ 2>&1 | FileCheck --check-prefix=CHECK-128P %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=256+ 2>&1 | FileCheck --check-prefix=CHECK-256P %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=512+ 2>&1 | FileCheck --check-prefix=CHECK-512P %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=1024+ 2>&1 | FileCheck --check-prefix=CHECK-1024P %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=2048+ 2>&1 | FileCheck --check-prefix=CHECK-2048P %s
! RUN: %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-SCALABLE %s

! CHECK-128: "-fc1"
! CHECK-128-SAME: "-mvscale-max=1" "-mvscale-min=1"
! CHECK-256: "-fc1"
! CHECK-256-SAME: "-mvscale-max=2" "-mvscale-min=2"
! CHECK-512: "-fc1"
! CHECK-512-SAME: "-mvscale-max=4" "-mvscale-min=4"
! CHECK-1024: "-fc1"
! CHECK-1024-SAME: "-mvscale-max=8" "-mvscale-min=8"
! CHECK-2048: "-fc1"
! CHECK-2048-SAME: "-mvscale-max=16" "-mvscale-min=16"

! CHECK-128P: "-fc1"
! CHECK-128P-SAME: "-mvscale-min=1"
! CHECK-128P-NOT: "-mvscale-max"
! CHECK-256P: "-fc1"
! CHECK-256P-SAME: "-mvscale-min=2"
! CHECK-256P-NOT: "-mvscale-max"
! CHECK-512P: "-fc1"
! CHECK-512P-SAME: "-mvscale-min=4"
! CHECK-512P-NOT: "-mvscale-max"
! CHECK-1024P: "-fc1"
! CHECK-1024P-SAME: "-mvscale-min=8"
! CHECK-1024P-NOT: "-mvscale-max"
! CHECK-2048P: "-fc1"
! CHECK-2048P-SAME: "-mvscale-min=16"
! CHECK-2048P-NOT: "-mvscale-max"
! CHECK-SCALABLE-NOT: "-mvscale-min=
! CHECK-SCALABLE-NOT: "-mvscale-max=

! Error out if an unsupported value is passed to -msve-vector-bits.
! -----------------------------------------------------------------------------
! RUN: not %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=64 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
! RUN: not %flang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
! RUN:  -msve-vector-bits=A 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s

! CHECK-BAD-VALUE-ERROR: error: unsupported argument '{{.*}}' to option '-msve-vector-bits='

