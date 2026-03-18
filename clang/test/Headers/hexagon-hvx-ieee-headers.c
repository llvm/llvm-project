// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -triple hexagon-unknown-elf -target-cpu hexagonv68  \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -target-feature +hvx-ieee-fp -emit-llvm %s -o - | FileCheck %s

// RUN: not %clang_cc1 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -triple hexagon-unknown-elf -target-cpu hexagonv68 \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

#include <hvx_hexagon_protos.h>
#include <hexagon_types.h>

HVX_Vector f(HVX_Vector v) {
  // CHECK-ERR: error: call to undeclared function 'Q6_Vhf_vabs_Vhf'
  // CHECK: call <32 x i32> @llvm.hexagon.V6.vabs.hf.128B
  return Q6_Vhf_vabs_Vhf(v);
}
