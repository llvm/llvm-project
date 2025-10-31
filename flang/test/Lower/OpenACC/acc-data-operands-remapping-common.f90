! Test remapping of common blocks appearing in OpenACC data directives.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test
  real :: x(100), y(100), overlap1(100), overlap2(100)
  equivalence (x(50), overlap1)
  equivalence (x(40), overlap2)
  common /comm/ x, y
  !$acc declare link(/comm/)
  !$acc parallel loop copyin(/comm/)
    do i = 1, 100
	   x(i) = overlap1(i)*2+ overlap2(i)
    enddo
end subroutine
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@comm_)
! CHECK:           %[[COPYIN_0:.*]] = acc.copyin varPtr(%[[ADDRESS_OF_0]] : !fir.ref<!fir.array<800xi8>>) -> !fir.ref<!fir.array<800xi8>> {name = "comm"}
! CHECK:           acc.parallel combined(loop) dataOperands(%[[COPYIN_0]] : !fir.ref<!fir.array<800xi8>>) {
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 196 : index
! CHECK:             %[[COORDINATE_OF_4:.*]] = fir.coordinate_of %[[COPYIN_0]], %{{.*}} : (!fir.ref<!fir.array<800xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_4:.*]] = fir.convert %[[COORDINATE_OF_4]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<100xf32>>
! CHECK:             %[[SHAPE_4:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:             %[[DECLARE_5:.*]]:2 = hlfir.declare %[[CONVERT_4]](%[[SHAPE_4]]) storage(%[[COPYIN_0]][196]) {uniq_name = "_QFtestEoverlap1"} : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>, !fir.ref<!fir.array<800xi8>>) -> (!fir.ptr<!fir.array<100xf32>>, !fir.ptr<!fir.array<100xf32>>)
! CHECK:             %[[CONSTANT_9:.*]] = arith.constant 156 : index
! CHECK:             %[[COORDINATE_OF_5:.*]] = fir.coordinate_of %[[COPYIN_0]], %{{.*}} : (!fir.ref<!fir.array<800xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_5:.*]] = fir.convert %[[COORDINATE_OF_5]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<100xf32>>
! CHECK:             %[[SHAPE_5:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:             %[[DECLARE_6:.*]]:2 = hlfir.declare %[[CONVERT_5]](%[[SHAPE_5]]) storage(%[[COPYIN_0]][156]) {uniq_name = "_QFtestEoverlap2"} : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>, !fir.ref<!fir.array<800xi8>>) -> (!fir.ptr<!fir.array<100xf32>>, !fir.ptr<!fir.array<100xf32>>)
! CHECK:             %[[CONSTANT_10:.*]] = arith.constant 0 : index
! CHECK:             %[[COORDINATE_OF_6:.*]] = fir.coordinate_of %[[COPYIN_0]], %{{.*}} : (!fir.ref<!fir.array<800xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_6:.*]] = fir.convert %[[COORDINATE_OF_6]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<100xf32>>
! CHECK:             %[[SHAPE_6:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:             %[[DECLARE_7:.*]]:2 = hlfir.declare %[[CONVERT_6]](%[[SHAPE_6]]) storage(%[[COPYIN_0]][0]) {uniq_name = "_QFtestEx"} : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>, !fir.ref<!fir.array<800xi8>>) -> (!fir.ptr<!fir.array<100xf32>>, !fir.ptr<!fir.array<100xf32>>)
! CHECK:             %[[CONSTANT_11:.*]] = arith.constant 400 : index
! CHECK:             %[[COORDINATE_OF_7:.*]] = fir.coordinate_of %[[COPYIN_0]], %{{.*}} : (!fir.ref<!fir.array<800xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_7:.*]] = fir.convert %[[COORDINATE_OF_7]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:             %[[SHAPE_7:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:             %[[DECLARE_8:.*]]:2 = hlfir.declare %[[CONVERT_7]](%[[SHAPE_7]]) storage(%[[COPYIN_0]][400]) {uniq_name = "_QFtestEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.ref<!fir.array<800xi8>>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:             acc.loop combined(parallel)
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_5]]#0
! CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_6]]#0
! CHECK:               %[[DESIGNATE_2:.*]] = hlfir.designate %[[DECLARE_7]]#0
