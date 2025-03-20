! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine sectionsReduction(x)
  real, dimension(:) :: x

  !$omp parallel
    !$omp sections reduction(+:x)
        x = x + 1
      !$omp section
        x = x + 2
    !$omp end sections
  !$omp end parallel
end subroutine


! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_Uxf32 : !fir.ref<!fir.box<!fir.array<?xf32>>> alloc {
! [...]
! CHECK:           omp.yield
! CHECK-LABEL:   } init {
! [...]
! CHECK:           omp.yield
! CHECK-LABEL:   } combiner {
! [...]
! CHECK:           omp.yield
! CHECK-LABEL:   }  cleanup {
! [...]
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QPsectionsreduction(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:             fir.store %[[VAL_2]]#1 to %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:             omp.sections reduction(byref @add_reduction_byref_box_Uxf32 %[[VAL_3]] -> %[[VAL_4:.*]] : !fir.ref<!fir.box<!fir.array<?xf32>>>) {
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_5:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>):
! [...]
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_23:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>):
! [...]
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

