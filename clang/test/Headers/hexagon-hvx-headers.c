// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -DDIRECT \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -x c++ \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length64b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-64 %s

#ifdef DIRECT
#include <hvx_hexagon_protos.h>
#else
#include <hexagon_protos.h>
#endif
#include <hexagon_types.h>

// expected-no-diagnostics

void test_hvx_protos(float a, unsigned int b) {
  HVX_VectorPair c;
  // CHECK-64: call <32 x i32> @llvm.hexagon.V6.v6mpyhubs10
  // CHECK:    call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.128B
  c = Q6_Ww_v6mpy_WubWbI_h(c, c, 2);
}
