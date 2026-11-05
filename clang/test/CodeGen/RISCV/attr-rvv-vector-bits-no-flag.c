// Verify that riscv_rvv_vector_bits typedefs work without -mrvv-vector-bits=,
// using only the ISA's Zvl*b extension as the lower bound. The fixed<->scalable
// conversion IR landmarks must be present.

// zvl128b, no -mrvv-vector-bits=:
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   -O1 -emit-llvm -o - %s | FileCheck %s

// The fixed<->scalable conversions fold away at -O1 now that both sides use
// the same LMUL, so check the landmarks on unoptimized IR.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   -O1 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=RAW

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

typedef vint32m1_t m1_64 __attribute__((riscv_rvv_vector_bits(64)));

// Fixed -> scalable conversion uses insert into poison. m1_64 keeps LMUL 1, so
// the ABI type already matches vint32m1_t and the conversion folds away.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @widen
// RAW: call <vscale x 2 x i32> @llvm.vector.insert.nxv2i32.v2i32(<vscale x 2 x i32> poison
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @widen
// CHECK-NEXT: entry:
// CHECK-NEXT: ret <vscale x 2 x i32> %x.coerce
vint32m1_t widen(m1_64 x) { return x; }

// Scalable -> fixed conversion uses extract.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @narrow
// RAW: call <2 x i32> @llvm.vector.extract.v2i32.nxv2i32(
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @narrow
// CHECK-NEXT: entry:
// CHECK-NEXT: ret <vscale x 2 x i32> %x
m1_64 narrow(vint32m1_t x) { return x; }
