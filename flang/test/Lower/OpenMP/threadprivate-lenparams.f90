! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Regression test for https://github.com/llvm/llvm-project/issues/108136

character(:), pointer :: c
character(2), pointer :: c2
!$omp threadprivate(c, c2)
end

! CHECK-LABEL:   fir.global internal @_QFEc : !fir.box<!fir.ptr<!fir.char<1,?>>> {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_0]] typeparams %[[VAL_1]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           fir.has_value %[[VAL_2]] : !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:         }

! CHECK-LABEL:   fir.global internal @_QFEc2 : !fir.box<!fir.ptr<!fir.char<1,2>>> {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,2>>
! CHECK:           %[[VAL_1:.*]] = fir.embox %[[VAL_0]] : (!fir.ptr<!fir.char<1,2>>) -> !fir.box<!fir.ptr<!fir.char<1,2>>>
! CHECK:           fir.has_value %[[VAL_1]] : !fir.box<!fir.ptr<!fir.char<1,2>>>
! CHECK:         }

