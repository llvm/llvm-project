// -----------------------------------------------------------------------------
// Tests for the -msve-vector-bits flag
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-128 %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=256 2>&1 | FileCheck --check-prefix=CHECK-256 %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=512 2>&1 | FileCheck --check-prefix=CHECK-512 %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=1024 2>&1 | FileCheck --check-prefix=CHECK-1024 %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=2048 2>&1 | FileCheck --check-prefix=CHECK-2048 %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-SCALABLE %s

// CHECK-128: "-msve-vector-bits=128"
// CHECK-256: "-msve-vector-bits=256"
// CHECK-512: "-msve-vector-bits=512"
// CHECK-1024: "-msve-vector-bits=1024"
// CHECK-2048: "-msve-vector-bits=2048"
// CHECK-SCALABLE-NOT: "-msve-vector-bits=

// Error out if an unsupported value is passed to -msve-vector-bits.
// -----------------------------------------------------------------------------
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=64 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: %clang -c %s -### -target aarch64-none-linux-gnu -march=armv8-a+sve \
// RUN:  -msve-vector-bits=A 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s

// CHECK-BAD-VALUE-ERROR: error: unsupported argument '{{.*}}' to option 'msve-vector-bits='

// Error if using attribute without -msve-vector-bits
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null -target aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s
// RUN: not %clang -c %s -o /dev/null -target aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve -msve-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s

typedef __SVInt32_t svint32_t;
typedef svint32_t noflag __attribute__((arm_sve_vector_bits(256)));

// CHECK-NO-FLAG-ERROR: error: 'arm_sve_vector_bits' is only supported when '-msve-vector-bits=<bits>' is specified with a value of 128, 256, 512, 1024 or 2048

// Error if attribute vector size != -msve-vector-bits
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null -target aarch64-none-linux-gnu \
// RUN:  -march=armv8-a+sve -msve-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-BAD-VECTOR-SIZE-ERROR %s

typedef svint32_t bad_vector_size __attribute__((arm_sve_vector_bits(256)));

// CHECK-BAD-VECTOR-SIZE-ERROR: error: invalid SVE vector size '256', must match value set by '-msve-vector-bits' ('128')
