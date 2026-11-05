// Verify implicit conversion semantics between a shorter
// riscv_rvv_vector_bits typedef and the underlying sizeless RVV type.
//
// VLEN=128 (vscale=2): vint32m1_t = <vscale x 2 x i32> (4 lanes total).
// A shorter typedef of vint32m1_t keeps LMUL 1, so it ABI-coerces to
// <vscale x 2 x i32> as well and only differs in how many lanes carry
// defined data:
//   fixed_i32x2 (bits(64)) declares 2 of the 4 i32 lanes.
//   fixed_i32x1 (bits(32)) declares 1 of the 4 i32 lanes.
//
// RAW checks the conversions Clang emits: llvm.vector.extract to truncate
// down to the declared lanes and llvm.vector.insert into poison to widen back
// with garbage. OPT checks that, because both sides now agree on LMUL, the
// round trips fold away entirely instead of leaving a change of register
// group behind.

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 \
// RUN:   -O1 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=RAW

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 \
// RUN:   -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

typedef vint32m1_t fixed_i32x4 __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32m1_t fixed_i32x2 __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t fixed_i32x1 __attribute__((riscv_rvv_vector_bits(32)));

vint32m1_t sizeless_callee(vint32m1_t x);

// fixed_i32x2 -> vint32m1_t: garbage-fill the lanes the typedef does not
// declare.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @widen(
// RAW-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
// RAW:         [[T0:%.*]] = call <2 x i32> @llvm.vector.extract.v2i32.nxv2i32(<vscale x 2 x i32> %x.coerce, i64 0)
// RAW:         call <vscale x 2 x i32> @llvm.vector.insert.nxv2i32.v2i32(<vscale x 2 x i32> poison, <2 x i32> {{.*}}, i64 0)
//
// OPT-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @widen(
// OPT-NEXT:  entry:
// OPT-NEXT:    ret <vscale x 2 x i32> %x.coerce
vint32m1_t widen(fixed_i32x2 x) { return x; }

// vint32m1_t -> fixed_i32x2: truncate to the first 2 lanes.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @narrow(
// RAW:         [[T0:%.*]] = call <2 x i32> @llvm.vector.extract.v2i32.nxv2i32(<vscale x 2 x i32> {{.*}}, i64 0)
//
// OPT-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @narrow(
// OPT-NEXT:  entry:
// OPT-NEXT:    ret <vscale x 2 x i32> %x
fixed_i32x2 narrow(vint32m1_t x) { return x; }

// vint32m1_t -> fixed_i32x1: truncate to the first lane, still returned in a
// whole LMUL=1 register.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @narrow_to_1(
// RAW:         call <1 x i32> @llvm.vector.extract.v1i32.nxv2i32(<vscale x 2 x i32> {{.*}}, i64 0)
// RAW:         call <vscale x 2 x i32> @llvm.vector.insert.nxv2i32.v1i32(<vscale x 2 x i32> poison, <1 x i32> {{.*}}, i64 0)
fixed_i32x1 narrow_to_1(vint32m1_t x) { return x; }

// fixed_i32x1 -> vint32m1_t: 1-lane garbage fill.
// RAW-LABEL: define dso_local <vscale x 2 x i32> @widen_from_1(
// RAW:         call <1 x i32> @llvm.vector.extract.v1i32.nxv2i32(<vscale x 2 x i32> %x.coerce, i64 0)
// RAW:         call <vscale x 2 x i32> @llvm.vector.insert.nxv2i32.v1i32(<vscale x 2 x i32> poison, <1 x i32> {{.*}}, i64 0)
vint32m1_t widen_from_1(fixed_i32x1 x) { return x; }

// Round-trip is a no-op after optimization: coerce in and out at the same
// scalable type.
// OPT-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @round_trip(
// OPT-NEXT:  entry:
// OPT-NEXT:    ret <vscale x 2 x i32> %x.coerce
fixed_i32x2 round_trip(fixed_i32x2 x) {
  vint32m1_t s = x;
  return s;
}

// Cross-call: passing a shorter fixed type to a sizeless callee needs no
// register-group change at all, since both are LMUL 1.
// OPT-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @pass_short(
// OPT-NEXT:  entry:
// OPT-NEXT:    [[C:%.*]] = {{.*}}call <vscale x 2 x i32> @sizeless_callee(<vscale x 2 x i32> %x.coerce)
// OPT-NEXT:    ret <vscale x 2 x i32> [[C]]
vint32m1_t pass_short(fixed_i32x2 x) { return sizeless_callee(x); }

// Full-size case: unchanged from the strict path.
// OPT-LABEL: define dso_local noundef <vscale x 2 x i32> @widen_full(
// OPT-NEXT:  entry:
// OPT-NEXT:    ret <vscale x 2 x i32> %x.coerce
vint32m1_t widen_full(fixed_i32x4 x) { return x; }
