! Test lowering of firstprivate on derived type with allocatable components.
! The runtime is called to handled the deep copy of the allocatable components.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefix=FIR-CHECK

module m_firstprivate_derived_alloc_comp
 type point
   real, allocatable :: x(:)
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

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_alloc_compTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>):
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "acc.private.init"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:           acc.yield %[[VAL_2]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
!
! CHECK:         } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:           acc.terminator
! CHECK:         }
!
! CHECK-LABEL:   func.func @_QMm_firstprivate_derived_alloc_compPtest(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEi"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEn"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = acc.firstprivate varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> {name = "a"}
! CHECK:           acc.parallel combined(loop) firstprivate(@firstprivatization_ref_rec__QMm_firstprivate_derived_alloc_compTpoint -> %[[VAL_6]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
! CHECK:             %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_12:.*]] = acc.private varPtr(%[[VAL_3]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:             acc.loop combined(parallel) private(@privatization_ref_i32 -> %[[VAL_12]] : !fir.ref<i32>) control(%[[VAL_14:.*]] : i32) = (%[[VAL_9]] : i32) to (%[[VAL_10]] : i32)  step (%[[VAL_11]] : i32) {
! CHECK:               %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QMm_firstprivate_derived_alloc_compFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_8]]#0{"x"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:               %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:               %[[VAL_18:.*]] = arith.constant 10 : index
! CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_19]] : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! FIR-CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_alloc_compTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_alloc_compTpoint{x:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> init {
! FIR-CHECK:   } copy {
! FIR-CHECK:           fir.call @_FortranAAssignTemporary(
! FIR-CHECK:           acc.terminator
! FIR-CHECK:         }
