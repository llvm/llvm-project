! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module poly
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: proc => proc_p1
  end type

  type, extends(p1) :: p2
    integer :: c
  contains
    procedure :: proc => proc_p2
  end type

contains

  subroutine proc_p1(this)
    class(p1) :: this
    print*, 'call proc2_p1'
  end subroutine

  subroutine proc_p2(this)
    class(p2) :: this
    print*, 'call proc2_p2'
  end subroutine


! ------------------------------------------------------------------------------
! Test lowering of ALLOCATE statement for polymoprhic pointer
! ------------------------------------------------------------------------------

  subroutine test_pointer()
    class(p1), pointer :: p
    class(p1), allocatable, target :: c1, c2
    class(p1), pointer :: pa(:)
    class(p1), allocatable, target, dimension(:) :: c3, c4
    integer :: i

    allocate(p1::c1)
    allocate(p2::c2)
    allocate(p1::c3(2))
    allocate(p2::c4(4))

    p => c1
    call p%proc()

    p => c2
    call p%proc()

    p => c3(1)
    call p%proc()

    p => c4(2)
    call p%proc()

    pa => c3
    do i = 1, 2
      call pa(i)%proc()
    end do

    pa => c4
    do i = 1, 4
      call pa(i)%proc()
    end do

    pa => c4(2:4)
    do i = 1, 2
      call pa(i)%proc()
    end do

    deallocate(c1)
    deallocate(c2)
    deallocate(c3)
    deallocate(c4)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_pointer()
! CHECK-DAG: %[[C1_ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c1", fir.target, uniq_name = "_QMpolyFtest_pointerEc1"}
! CHECK-DAG: %[[C2_ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c2", fir.target, uniq_name = "_QMpolyFtest_pointerEc2"}
! CHECK-DAG: %[[C3_ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c3", fir.target, uniq_name = "_QMpolyFtest_pointerEc3"}
! CHECK-DAG: %[[C4_ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c4", fir.target, uniq_name = "_QMpolyFtest_pointerEc4"}
! CHECK-DAG: %[[P_ALLOCA:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_pointerEp"}
! CHECK-DAG: %[[PA_ALLOCA:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "pa", uniq_name = "_QMpolyFtest_pointerEpa"}
! CHECK-DAG: %[[C1:.*]]:2 = hlfir.declare %[[C1_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMpolyFtest_pointerEc1"}
! CHECK-DAG: %[[C2:.*]]:2 = hlfir.declare %[[C2_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMpolyFtest_pointerEc2"}
! CHECK-DAG: %[[C3:.*]]:2 = hlfir.declare %[[C3_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMpolyFtest_pointerEc3"}
! CHECK-DAG: %[[C4:.*]]:2 = hlfir.declare %[[C4_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMpolyFtest_pointerEc4"}
! CHECK-DAG: %[[P:.*]]:2 = hlfir.declare %[[P_ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolyFtest_pointerEp"}
! CHECK-DAG: %[[PA:.*]]:2 = hlfir.declare %[[PA_ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolyFtest_pointerEpa"}

! p => c1
! CHECK: %[[C1_LOAD:.*]] = fir.load %[[C1]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[C1_REBOX]] to %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! p => c2
! CHECK: %[[C2_LOAD:.*]] = fir.load %[[C2]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[C2_REBOX]] to %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! p => c3(1)
! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_ELEM:.*]] = hlfir.designate %[[C3_LOAD]] (%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_REBOX:.*]] = fir.rebox %[[C3_ELEM]] : (!fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[C3_REBOX]] to %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! p => c4(2)
! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_ELEM:.*]] = hlfir.designate %[[C4_LOAD]] (%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_REBOX:.*]] = fir.rebox %[[C4_ELEM]] : (!fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[C4_REBOX]] to %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! pa => c3
! CHECK: %[[C3_LOAD_A:.*]] = fir.load %[[C3]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_REBOX_A:.*]] = fir.rebox %[[C3_LOAD_A]](%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.store %[[C3_REBOX_A]] to %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD:.*]] = fir.load %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_ELEM:.*]] = hlfir.designate %[[PA_LOAD]] (%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_ELEM]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_ELEM]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! pa => c4
! CHECK: %[[C4_LOAD_A:.*]] = fir.load %[[C4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_REBOX_A:.*]] = fir.rebox %[[C4_LOAD_A]](%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.store %[[C4_REBOX_A]] to %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD2:.*]] = fir.load %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_ELEM2:.*]] = hlfir.designate %[[PA_LOAD2]] (%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_ELEM2]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_ELEM2]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! pa => c4(2:4)
! CHECK: %[[C4_LOAD_S:.*]] = fir.load %[[C4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[SLICE:.*]] = hlfir.designate %[[C4_LOAD_S]] (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index, index, index, !fir.shape<1>) -> !fir.class<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: %[[SLICE_REBOX:.*]] = fir.rebox %[[SLICE]] : (!fir.class<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.ptr<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[SLICE_CONV:.*]] = fir.convert %[[SLICE_REBOX]] : (!fir.class<!fir.ptr<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.store %[[SLICE_CONV]] to %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD3:.*]] = fir.load %[[PA]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_ELEM3:.*]] = hlfir.designate %[[PA_LOAD3]] (%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_ELEM3]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_ELEM3]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

end module

program test_pointer_association
  use poly
  call test_pointer()
end
