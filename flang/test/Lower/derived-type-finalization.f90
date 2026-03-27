! Test derived type finalization
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Missing tests:
! - finalization within BLOCK construct

module derived_type_finalization

  type :: t1
    integer :: a
  contains
    final :: t1_final
    final :: t1_final_1r
  end type

  type :: t2
    integer, allocatable, dimension(:) :: a
  contains
    final :: t2_final
  end type

  type :: t3
    type(t2) :: t
  end type

  type t4
  contains
    final :: t4_final
  end type

contains

  subroutine t1_final(this)
    type(t1) :: this
  end subroutine

  subroutine t1_final_1r(this)
    type(t1) :: this(:)
  end subroutine

  subroutine t2_final(this)
    type(t2) :: this
  end subroutine

  ! 7.5.6.3 point 1. Finalization of LHS.
  subroutine test_lhs()
    type(t1) :: lhs, rhs
    lhs = rhs
  end subroutine

  subroutine test_lhs_allocatable()
    type(t1), allocatable :: lhs
    type(t1) :: rhs
    lhs = rhs
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_lhs() {
! CHECK: %[[LHS:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}> {bindc_name = "lhs", uniq_name = "_QMderived_type_finalizationFtest_lhsElhs"}
! CHECK: %[[LHS_DECL:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK: %[[RHS:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}> {bindc_name = "rhs", uniq_name = "_QMderived_type_finalizationFtest_lhsErhs"}
! CHECK: %[[RHS_DECL:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK: hlfir.assign %[[RHS_DECL]]#0 to %[[LHS_DECL]]#0
! CHECK: fir.call @_FortranADestroy
! CHECK: fir.call @_FortranADestroy

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_lhs_allocatable() {
! CHECK: %[[LHS:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>> {bindc_name = "lhs", uniq_name = "_QMderived_type_finalizationFtest_lhs_allocatableElhs"}
! CHECK: %[[LHS_DECL:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK: %[[RHS:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}> {bindc_name = "rhs", uniq_name = "_QMderived_type_finalizationFtest_lhs_allocatableErhs"}
! CHECK: %[[RHS_DECL:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK: hlfir.assign %[[RHS_DECL]]#0 to %[[LHS_DECL]]#0 realloc
! CHECK: fir.call @_FortranADestroy

  ! 7.5.6.3 point 2. Finalization on explicit deallocation.
  subroutine test_deallocate()
    type(t1), allocatable :: t
    allocate(t)
    deallocate(t)
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_deallocate() {
! CHECK: %[[T:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>>
! CHECK: %[[T_DECL:.*]]:2 = hlfir.declare %[[T]]
! CHECK: fir.call @_FortranAAllocatableAllocate
! CHECK: fir.call @_FortranAAllocatableDeallocate

  ! 7.5.6.3 point 2. Finalization of disassociated target.
  subroutine test_target_finalization()
    type(t1), pointer :: p
    allocate(p, source=t1(a=2))
    deallocate(p)
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_target_finalization() {
! CHECK: %[[P:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>>
! CHECK: %[[P_DECL:.*]]:2 = hlfir.declare %[[P]]
! CHECK: fir.call @_FortranAPointerAllocateSource
! CHECK: fir.call @_FortranAPointerDeallocate

  ! 7.5.6.3 point 3. Finalize on END.
  subroutine test_end_finalization()
    type(t1) :: t
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_end_finalization() {
! CHECK: %[[T:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}>
! CHECK: %[[T_DECL:.*]]:2 = hlfir.declare %[[T]]
! CHECK: fir.call @_FortranADestroy

  ! test with multiple return.
  subroutine test_end_finalization2(a)
    type(t1) :: t
    logical :: a
    if (a) return
    t%a = 10
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_end_finalization2(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "a"}) {
! CHECK:   %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK:   %[[T:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}> {bindc_name = "t", uniq_name = "_QMderived_type_finalizationFtest_end_finalization2Et"}
! CHECK:   %[[T_DECL:.*]]:2 = hlfir.declare %[[T]]
! CHECK:   cf.cond_br
! CHECK:   hlfir.assign
! CHECK:   fir.call @_FortranADestroy
! CHECK:   return
! CHECK: }

  function ret_type() result(ty)
    type(t1) :: ty
  end function

  ! 7.5.6.3 point 5. Finalization of a function reference on the RHS of an intrinsic assignment.
  subroutine test_fct_ref()
    type(t1), allocatable :: ty
    ty = ret_type()
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_fct_ref() {
! CHECK: %[[RES:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt1{a:i32}> {bindc_name = ".result"}
! CHECK: %[[TY:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>>
! CHECK: %[[TY_DECL:.*]]:2 = hlfir.declare %[[TY]]
! CHECK: %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES]]
! CHECK: %[[CALL_RES:.*]] = fir.call @_QMderived_type_finalizationPret_type()
! CHECK: fir.save_result %[[CALL_RES]] to %[[RES_DECL]]#0
! CHECK: hlfir.assign %[[RES_DECL]]#0 to %[[TY_DECL]]#0 realloc
! CHECK: fir.call @_FortranADestroy(%{{.*}})
! CHECK: return

  subroutine test_finalize_intent_out(t)
    type(t1), intent(out) :: t
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_finalize_intent_out(
! CHECK-SAME: %[[T:.*]]: !fir.ref<!fir.type<_QMderived_type_finalizationTt1{a:i32}>> {fir.bindc_name = "t"}) {
! CHECK: %[[T_DECL:.*]]:2 = hlfir.declare %[[T]]
! CHECK: fir.call @_FortranADestroy(%{{.*}})
! CHECK: return

  function get_t1(i)
    type(t1), pointer :: get_t1
    allocate(get_t1)
    get_t1%a = i
  end function

  subroutine test_nonpointer_function()
    print*, get_t1(20)
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_nonpointer_function() {
! CHECK: %[[TMP:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>> {bindc_name = ".result"}
! CHECK: %{{.*}} = fir.call @_FortranAioBeginExternalListOutput
! CHECK: %[[RES:.*]] = fir.call @_QMderived_type_finalizationPget_t1(%{{.*}})
! CHECK: fir.save_result %[[RES]] to %[[TMP]] : !fir.box<!fir.ptr<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMderived_type_finalizationTt1{a:i32}>>>>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputDerivedType
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK: %{{.*}} = fir.call @_FortranAioEndIoStatement
! CHECK: return

  subroutine test_avoid_double_finalization(a)
    type(t3), intent(inout) :: a
    type(t3)                :: b
    b = a
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_avoid_double_finalization(
! CHECK: %[[b:.*]] = fir.alloca !fir.type<_QMderived_type_finalizationTt3{t:!fir.type<_QMderived_type_finalizationTt2{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>}> {bindc_name = "b", uniq_name = "_QMderived_type_finalizationFtest_avoid_double_finalizationEb"}
! CHECK: %[[b_DECL:.*]]:2 = hlfir.declare %[[b]]
! CHECK: fir.copy %{{.*}} to %[[b_DECL]]#0
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK: hlfir.assign
! CHECK: fir.call @_FortranADestroy

  function no_func_ret_finalize() result(ty)
    type(t1) :: ty
    ty = t1(10)
  end function

! CHECK-LABEL: func.func @_QMderived_type_finalizationPno_func_ret_finalize() -> !fir.type<_QMderived_type_finalizationTt1{a:i32}> {
! CHECK: hlfir.assign
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK: return %{{.*}} : !fir.type<_QMderived_type_finalizationTt1{a:i32}>

  function copy(a) result(ty)
    class(t1), allocatable :: ty(:)
    integer, intent(in) :: a
    allocate(t1::ty(a))
    ty%a = 1
  end function

  subroutine test_avoid_double_free()
    class(*), allocatable :: up(:)
    allocate(up(10), source=copy(10))
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPtest_avoid_double_free() {
! CHECK: fir.call @_FortranAAllocatableAllocateSource(
! CHECK: fir.call @_FortranADestroy
! CHECK: fir.call @_FortranAAllocatableDeallocatePolymorphic

  subroutine t4_final(this)
    type(t4) :: this
  end subroutine

  subroutine local_t4()
    type(t4) :: t
  end subroutine

! CHECK-LABEL: func.func @_QMderived_type_finalizationPlocal_t4()
! CHECK: fir.call @_FortranADestroy(%{{.*}}) fastmath<contract> : (!fir.box<none>) -> ()

end module

program p
  use derived_type_finalization
  type(t1) :: t
end program

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "P"} {
! CHECK-NOT: fir.call @_FortranADestroy
