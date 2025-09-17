! Test lowering of firstprivate on derived type with user defined assignments,
! The user defined assignments should not be called when making firstprivate
! copies.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefix=FIR-CHECK

module m_firstprivate_derived_user_def
 type point
   real :: x, y, z
  contains
    procedure :: user_copy
    generic :: assignment(=) => user_copy
 end type point
 contains

 subroutine user_copy(lhs, rhs)
   class(point), intent(out) :: lhs
   class(point), intent(in) :: rhs
   print *, "hello, I am a side effect"
   lhs%x = rhs%x
   lhs%y = rhs%y
   lhs%z = rhs%z
 end subroutine

   subroutine test()
     type(point) :: a

     !$acc parallel loop firstprivate(a)
     do i = 1, n
      a%x = 1
     enddo
   end
 end module

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
!
! CHECK:         } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! CHECK:           acc.terminator
! CHECK:         }
!
! CHECK-LABEL:   func.func @_QMm_firstprivate_derived_user_defPtest() {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}> {bindc_name = "a", uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QMm_firstprivate_derived_user_defFtestEn"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = acc.firstprivate varPtr(%[[VAL_2]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>> {name = "a"}
! CHECK:           acc.parallel combined(loop) firstprivate(@firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint -> %[[VAL_7]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) {
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_11:.*]] = acc.private varPtr(%[[VAL_4]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:             %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             acc.loop combined(parallel) private(@privatization_ref_i32 -> %[[VAL_11]] : !fir.ref<i32>) control(%[[VAL_13:.*]] : i32) = (%[[VAL_8]] : i32) to (%[[VAL_9]] : i32)  step (%[[VAL_10]] : i32) {
! CHECK:               fir.store %[[VAL_13]] to %[[VAL_12]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_14:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               %[[VAL_15:.*]] = hlfir.designate %[[VAL_2]]#0{"x"}   : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! CHECK:               hlfir.assign %[[VAL_14]] to %[[VAL_15]] : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! FIR-CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>> init {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! FIR-CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! FIR-
! FIR-CHECK:        } copy {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! FIR-CHECK:           %[[VAL_2:.*]] = fir.field_index x, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], x : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_4:.*]] = fir.field_index x, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_1]], x : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]] : !fir.ref<f32>
! FIR-CHECK:           fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<f32>
! FIR-CHECK:           %[[VAL_7:.*]] = fir.field_index y, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_0]], y : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_9:.*]] = fir.field_index y, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_1]], y : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]] : !fir.ref<f32>
! FIR-CHECK:           fir.store %[[VAL_11]] to %[[VAL_10]] : !fir.ref<f32>
! FIR-CHECK:           %[[VAL_12:.*]] = fir.field_index z, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_0]], z : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_14:.*]] = fir.field_index z, !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_1]], z : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! FIR-CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
! FIR-CHECK:           fir.store %[[VAL_16]] to %[[VAL_15]] : !fir.ref<f32>
! FIR-CHECK:           acc.terminator
! FIR-CHECK:         }
