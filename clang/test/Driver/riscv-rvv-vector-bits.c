// -----------------------------------------------------------------------------
// Tests for the -mrvv-vector-bits flag
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-128 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=256 2>&1 | FileCheck --check-prefix=CHECK-256 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=512 2>&1 | FileCheck --check-prefix=CHECK-512 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=1024 2>&1 | FileCheck --check-prefix=CHECK-1024 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=2048 2>&1 | FileCheck --check-prefix=CHECK-2048 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-SCALABLE %s

// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv_zvl256b \
// RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-256 %s
// RUN: %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv_zvl512b \
// RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-512 %s

// CHECK-128: "-mvscale-max=2" "-mvscale-min=2"
// CHECK-256: "-mvscale-max=4" "-mvscale-min=4"
// CHECK-512: "-mvscale-max=8" "-mvscale-min=8"
// CHECK-1024: "-mvscale-max=16" "-mvscale-min=16"
// CHECK-2048: "-mvscale-max=32" "-mvscale-min=32"

// CHECK-SCALABLE-NOT: "-mvscale-min=
// CHECK-SCALABLE-NOT: "-mvscale-max=

// Error out if an unsupported value is passed to -mrvv-vector-bits.
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=16 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: not %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=A 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: not %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc_zve64x \
// RUN:  -mrvv-vector-bits=131072 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: not %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gc \
// RUN:  -mrvv-vector-bits=zvl 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s
// RUN: not %clang -c %s -### --target=riscv64-linux-gnu -march=rv64gcv \
// RUN:  -mrvv-vector-bits=64 2>&1 | FileCheck --check-prefix=CHECK-BAD-VALUE-ERROR %s

// CHECK-BAD-VALUE-ERROR: error: unsupported argument '{{.*}}' to option '-mrvv-vector-bits='

// Error if using attribute without -msve-vector-bits=<bits> or if using -msve-vector-bits=<bits>+ syntax
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null -target riscv64-linux-gnu \
// RUN:  -march=rv64gc_zve64x 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s
// RUN: not %clang -c %s -o /dev/null -target riscv64-linux-gnu \
// RUN:  -march=rv64gc_zve64x -mrvv-vector-bits=scalable 2>&1 | FileCheck --check-prefix=CHECK-NO-FLAG-ERROR %s

typedef __rvv_int32m1_t vint32m1_t;
typedef vint32m1_t noflag __attribute__((riscv_rvv_vector_bits(256)));

// CHECK-NO-FLAG-ERROR: error: 'riscv_rvv_vector_bits' is only supported when '-mrvv-vector-bits=<bits>' is specified with a value of "zvl" or a power 2 in the range [64,65536]

// Error if attribute vector size != -mrvv-vector-bits
// -----------------------------------------------------------------------------
// RUN: not %clang -c %s -o /dev/null -target riscv64-linux-gnu \
// RUN:  -march=rv64gc_zve64x -mrvv-vector-bits=128 2>&1 | FileCheck --check-prefix=CHECK-BAD-VECTOR-SIZE-ERROR %s

typedef vint32_t bad_vector_size __attribute__((riscv_rvv_vector_bits(256)));

// CHECK-BAD-VECTOR-SIZE-ERROR: error: invalid RVV vector size '256', expected size is '128' based on LMUL of type and '-mrvv-vector-bits'
