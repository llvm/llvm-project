! Downstream only: support for equivalence with threadprivate/copyprivate

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

subroutine s1
  equivalence (k(2),kk)
  common /com1/ k(2)
  !$omp threadprivate(/com1/)

  !$omp parallel
    k=0
    !$omp single
      k(1)=1
      kk=2
    !$omp end single copyprivate(/com1/)
  !$omp end parallel
end

! Check we can generate a copy function for !fir.ptr arguments
! CHECK-LABEL:   func.func private @_copy_ptr_2xi32(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ptr<!fir.array<2xi32>>,
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ptr<!fir.array<2xi32>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]](%[[SHAPE_0]]) {uniq_name = "_copy_ptr_2xi32_dst"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ARG1]](%[[SHAPE_0]]) {uniq_name = "_copy_ptr_2xi32_src"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)
! CHECK:           hlfir.assign %[[DECLARE_1]]#0 to %[[DECLARE_0]]#0 : !fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPs1() {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@com1_) : !fir.ref<!fir.array<8xi8>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[ADDRESS_OF_0]], %[[CONSTANT_0]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[COORDINATE_OF_0]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[CONVERT_0]](%[[SHAPE_0]]) storage(%[[ADDRESS_OF_0]][0]) {uniq_name = "_QFs1Ek"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)
! CHECK:           %[[THREADPRIVATE_0:.*]] = omp.threadprivate %[[ADDRESS_OF_0]] : !fir.ref<!fir.array<8xi8>> -> !fir.ref<!fir.array<8xi8>>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[COORDINATE_OF_1:.*]] = fir.coordinate_of %[[THREADPRIVATE_0]], %[[CONSTANT_2]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[COORDINATE_OF_1]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[CONVERT_1]](%[[SHAPE_1]]) storage(%[[THREADPRIVATE_0]][0]) {uniq_name = "_QFs1Ek"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
! CHECK:           %[[COORDINATE_OF_2:.*]] = fir.coordinate_of %[[THREADPRIVATE_0]], %[[CONSTANT_3]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[COORDINATE_OF_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[CONVERT_2]] storage(%[[THREADPRIVATE_0]][4]) {uniq_name = "_QFs1Ekk"} : (!fir.ptr<i32>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:           omp.parallel {
! CHECK:             %[[THREADPRIVATE_1:.*]] = omp.threadprivate %[[ADDRESS_OF_0]] : !fir.ref<!fir.array<8xi8>> -> !fir.ref<!fir.array<8xi8>>
! CHECK:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
! CHECK:             %[[COORDINATE_OF_3:.*]] = fir.coordinate_of %[[THREADPRIVATE_1]], %[[CONSTANT_4]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[COORDINATE_OF_3]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
! CHECK:             %[[SHAPE_2:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:             %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_3]](%[[SHAPE_2]]) storage(%[[THREADPRIVATE_1]][0]) {uniq_name = "_QFs1Ek"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)
! CHECK:             %[[CONSTANT_5:.*]] = arith.constant 4 : index

! Check that the equivalence'd value kk comes from omp.threadprivate not from a shared variable
! CHECK:             %[[COORDINATE_OF_4:.*]] = fir.coordinate_of %[[THREADPRIVATE_1]], %[[CONSTANT_5]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[CONVERT_4:.*]] = fir.convert %[[COORDINATE_OF_4]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:             %[[DECLARE_4:.*]]:2 = hlfir.declare %[[CONVERT_4]] storage(%[[THREADPRIVATE_1]][4]) {uniq_name = "_QFs1Ekk"} : (!fir.ptr<i32>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)

! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : i32
! CHECK:             hlfir.assign %[[CONSTANT_6]] to %[[DECLARE_3]]#0 : i32, !fir.ptr<!fir.array<2xi32>>
! CHECK:             omp.single copyprivate(%[[DECLARE_3]]#0 -> @_copy_ptr_2xi32 : !fir.ptr<!fir.array<2xi32>>) {
! CHECK:               %[[CONSTANT_7:.*]] = arith.constant 1 : i32
! CHECK:               %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_3]]#0 (%[[CONSTANT_8]])  : (!fir.ptr<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:               hlfir.assign %[[CONSTANT_7]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32>
! CHECK:               %[[CONSTANT_9:.*]] = arith.constant 2 : i32
! CHECK:               hlfir.assign %[[CONSTANT_9]] to %[[DECLARE_4]]#0 : i32, !fir.ptr<i32>
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
