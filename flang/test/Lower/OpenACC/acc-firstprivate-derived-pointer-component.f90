! Test lowering of firstprivate on derived type with pointer components.
! No deep copy must be done.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefix=FIR-CHECK

module m_firstprivate_derived_ptr_comp
 type point
   real, pointer :: x(:)
 end type point
 contains
   subroutine test(a)
     type(point) :: a

     !$acc parallel loop firstprivate(a)
     do i = 1, n
      a%x(10) = 1
     enddo
   end
 end module

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_ptr_compTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>):
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
!
! CHECK:         } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           acc.terminator
! CHECK:         }
!
! CHECK-LABEL:   func.func @_QMm_firstprivate_derived_ptr_compPtest(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEi"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEn"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = acc.firstprivate varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {name = "a"}
! CHECK:           acc.parallel combined(loop) firstprivate(@firstprivatization_ref_rec__QMm_firstprivate_derived_ptr_compTpoint -> %[[VAL_6]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) {
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_10:.*]] = acc.private varPtr(%[[VAL_3]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:             %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QMm_firstprivate_derived_ptr_compFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             acc.loop combined(parallel) private(@privatization_ref_i32 -> %[[VAL_10]] : !fir.ref<i32>) control(%[[VAL_12:.*]] : i32) = (%[[VAL_7]] : i32) to (%[[VAL_8]] : i32)  step (%[[VAL_9]] : i32) {
! CHECK:               fir.store %[[VAL_12]] to %[[VAL_11]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_13:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               %[[VAL_14:.*]] = hlfir.designate %[[VAL_1]]#0{"x"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:               %[[VAL_16:.*]] = arith.constant 10 : index
! CHECK:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_15]] (%[[VAL_16]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:               hlfir.assign %[[VAL_13]] to %[[VAL_17]] : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! FIR-CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_ptr_compTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> init {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>):
! FIR-CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
!
! FIR-CHECK-LABEL:   } copy {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>):
! FIR-CHECK:           %[[VAL_2:.*]] = fir.field_index x, !fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! FIR-CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], x : (!fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! FIR-CHECK:           %[[VAL_4:.*]] = fir.field_index x, !fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! FIR-CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_1]], x : (!fir.ref<!fir.type<_QMm_firstprivate_derived_ptr_compTpoint{x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! FIR-CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! FIR-CHECK:           fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! FIR-CHECK:           acc.terminator
! FIR-CHECK:         }
