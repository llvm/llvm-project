// -----------------------------------------------------------------------------
// Tests for the -msve-vector-bits flag
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-128 %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=256 2>&1 | FileCheck --check-prefix=CHECK-256 %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=512 2>&1 | FileCheck --check-prefix=CHECK-512 %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=1024 2>&1 | FileCheck --check-prefix=CHECK-1024 %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=2048 2>&1 | FileCheck --check-prefix=CHECK-2048 %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=128+ 2>&1 | FileCheck --check-prefix=CHECK-128P %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=256+ 2>&1 | FileCheck --check-prefix=CHECK-256P %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=512+ 2>&1 | FileCheck --check-prefix=CHECK-512P %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=1024+ 2>&1 | FileCheck --check-prefix=CHECK-1024P %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=2048+ 2>&1 | FileCheck --check-prefix=CHECK-2048P %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-SCALABLE %s
// RUN: %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve+sme \
// RUN:  -msve-streaming-vector-bits=128 2>&1 | FileCheck --check-prefix=STREAMING-128 %s

// CHECK-128: "-mvscale-max=1" "-mvscale-min=1"
// CHECK-256: "-mvscale-max=2" "-mvscale-min=2"
// CHECK-512: "-mvscale-max=4" "-mvscale-min=4"
// CHECK-1024: "-mvscale-max=8" "-mvscale-min=8"
// CHECK-2048: "-mvscale-max=16" "-mvscale-min=16"

// CHECK-128P: "-mvscale-min=1"
// CHECK-128P-NOT: "-mvscale-max"
// CHECK-256P: "-mvscale-min=2"
// CHECK-256P-NOT: "-mvscale-max"
// CHECK-512P: "-mvscale-min=4"
// CHECK-512P-NOT: "-mvscale-max"
// CHECK-1024P: "-mvscale-min=8"
// CHECK-1024P-NOT: "-mvscale-max"
// CHECK-2048P: "-mvscale-min=16"
// CHECK-2048P-NOT: "-mvscale-max"
// CHECK-SCALABLE-NOT: "-mvscale-min=
// CHECK-SCALABLE-NOT: "-mvscale-max=

// STREAMING-128: "-mvscale-streaming-max=1" "-mvscale-streaming-min=1"

// Error out if an unsupported value is passed to -msve-vector-bits.
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=64 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: not %clang -c %s -### --target=aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=A 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s

// CHECK-BAD-VALUE-ERROR: error: unsupported argument '{{.*}}' to option '-msve-vector-bits='

// Error if using attribute without -msve-vector-bits=<bits> or if using -msve-vector-bits=<bits>+ syntax
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null --target=aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s
// RUN: not %clang -c %s -o /dev/null --target=aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve -msve-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s
// RUN: not %clang -c %s -o /dev/null --target=aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve -msve-vector-bits=256+ 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s

typedef __SVInt32_t svint32_t;
typedef svint32_t noflag __attribute__((arm_sve_vector_bits(256)));

// CHECK-NO-FLAG-ERROR: error: 'arm_sve_vector_bits' is only supported when '-msve-vector-bits=<bits>' is specified with a value of 128, 256, 512, 1024 or 2048

// Error if attribute vector size != -msve-vector-bits
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null --target=aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve -msve-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-BAD-VECTOR-SIZE-ERROR %s

typedef svint32_t bad_vector_size __attribute__((arm_sve_vector_bits(256)));

// CHECK-BAD-VECTOR-SIZE-ERROR: error: invalid SVE vector size '256', must match value set by '-msve-vector-bits' ('128')
