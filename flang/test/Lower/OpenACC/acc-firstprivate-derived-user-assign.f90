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
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! CHECK:           acc.yield %[[VAL_1]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
!
! CHECK:         } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! CHECK:           acc.terminator
! CHECK:         }
!
! CHECK-LABEL:   func.func @_QMm_firstprivate_derived_user_defPtest() {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}> <{bindc_name = "a", uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"}>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"}>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 <{bindc_name = "n", uniq_name = "_QMm_firstprivate_derived_user_defFtestEn"}>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = acc.firstprivate varPtr(%[[VAL_2]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) recipe(@firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint) name("a") -> !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! CHECK:           acc.parallel combined(loop) firstprivate(%[[VAL_7]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) {
! CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"} : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>)
! CHECK:             %[[FP_A_LOOP:.*]] = acc.firstprivate varPtr(%[[VAL_8]]#0 : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) recipe(@firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint) implicit(true) name("a") -> !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_12:.*]] = acc.private varPtr(%[[VAL_4]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i") -> !fir.ref<i32>
! CHECK:             acc.loop combined(parallel) private(%[[VAL_12]] : !fir.ref<i32>) firstprivate(%[[FP_A_LOOP]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) control(%[[VAL_14:.*]] : i32) = (%[[VAL_9]] : i32) to (%[[VAL_10]] : i32)  step (%[[VAL_11]] : i32) {
! CHECK:               %[[VAL_8B:.*]]:2 = hlfir.declare %[[FP_A_LOOP]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEa"}
  ! CHECK:             %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QMm_firstprivate_derived_user_defFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_8B]]#0{"x"}   : (!fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>) -> !fir.ref<f32>
! CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             } inclusiveUpperbound(array<i1: true>) independent
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! FIR-CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_rec__QMm_firstprivate_derived_user_defTpoint : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>> init {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! FIR-CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>
! FIR-CHECK:           acc.yield %[[VAL_1]] : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! FIR-CHECK:        } copy {
! FIR-CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>):
! FIR-CHECK:           fir.copy %[[VAL_0]] to %[[VAL_1]] no_overlap : !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QMm_firstprivate_derived_user_defTpoint{x:f32,y:f32,z:f32}>>
! FIR-CHECK:           acc.terminator
! FIR-CHECK:         }
