! RUN: bbc --emit-fir -hlfir=false %s -o - | FileCheck %s

! Test generation of dense attributed global array. Also, make sure there are
! no dead ssa assignments.
module mm
  integer, parameter :: qq(3) = [(i,i=51,53)]
end
subroutine ss
  use mm
  n = qq(3)
end
!CHECK-NOT: %{{.*}} = fir.undefined !fir.array<3xi32>
!CHECK-NOT: %{{.*}} = arith.constant %{{.*}} : index
!CHECK-NOT: %{{.*}} = arith.constant %{{.*}} : i32
!CHECK-NOT: %{{.*}} = fir.insert_value %{{.*}}, %{{.*}}, [%{{.*}} : index] : (!fir.array<3xi32>, i32) -> !fir.array<3xi32>
!CHECK: fir.global @_QMmmECqq(dense<[51, 52, 53]> : tensor<3xi32>) constant : !fir.array<3xi32>
!CHECK: func @_QPss() {
!CHECK:  %[[a0:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFssEn"}
!CHECK:  %[[c0:.*]] = arith.constant 53 : i32
!CHECK:  fir.store %[[c0]] to %[[a0]] : !fir.ref<i32>
!CHECK:  return
!CHECK: }
