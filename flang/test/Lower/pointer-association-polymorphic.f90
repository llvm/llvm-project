! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

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
! CHECK-DAG: %[[C1_DESC:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c1", fir.target, uniq_name = "_QMpolyFtest_pointerEc1"}
! CHECK-DAG: %[[C2_DESC:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c2", fir.target, uniq_name = "_QMpolyFtest_pointerEc2"}
! CHECK-DAG: %[[C3_DESC:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c3", fir.target, uniq_name = "_QMpolyFtest_pointerEc3"}
! CHECK-DAG: %[[C4_DESC:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c4", fir.target, uniq_name = "_QMpolyFtest_pointerEc4"}
! CHECK-DAG: %[[P_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_pointerEp"}
! CHECK-DAG: %[[PA_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "pa", uniq_name = "_QMpolyFtest_pointerEpa"}

! CHECK: %[[C1_DESC_LOAD:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_CONV:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_DESC_CONV:.*]] = fir.convert %[[C1_DESC_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[P_CONV]], %[[C1_DESC_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
! CHECK: %[[P_DESC_LOAD:.*]] = fir.load %[[P_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_DESC_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C2_DESC_LOAD:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_CONV:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C2_DESC_CONV:.*]] = fir.convert %[[C2_DESC_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[P_CONV]], %[[C2_DESC_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
! CHECK: %[[P_DESC_LOAD:.*]] = fir.load %[[P_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_DESC_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[C3_DIMS:.*]]:3 = fir.box_dims %[[C3_LOAD]], %[[C0]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : i64
! CHECK: %[[LB:.*]] = fir.convert %[[C3_DIMS]]#0 : (index) -> i64
! CHECK: %[[IDX:.*]] = arith.subi %[[C1]], %[[LB]] : i64
! CHECK: %[[C3_COORD:.*]] = fir.coordinate_of %[[C3_LOAD]], %[[IDX]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_EMBOX:.*]] = fir.embox %[[C3_COORD]] source_box %[[C3_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[P_CONV:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C3_EMBOX_CONV:.*]] = fir.convert %[[C3_EMBOX]] : (!fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[P_CONV]], %[[C3_EMBOX_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
! CHECK: %[[P_DESC_LOAD:.*]] = fir.load %[[P_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_DESC_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[C4_DIMS:.*]]:3 = fir.box_dims %[[C4_LOAD]], %[[C0]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[C2:.*]] = arith.constant 2 : i64
! CHECK: %[[LB:.*]] = fir.convert %[[C4_DIMS]]#0 : (index) -> i64
! CHECK: %[[IDX:.*]] = arith.subi %[[C2]], %[[LB]] : i64
! CHECK: %[[C4_COORD:.*]] = fir.coordinate_of %[[C4_LOAD]], %[[IDX]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_EMBOX:.*]] = fir.embox %[[C4_COORD]] source_box %[[C4_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[P_CONV:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C4_EMBOX_CONV:.*]] = fir.convert %[[C4_EMBOX]] : (!fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[P_CONV]], %[[C4_EMBOX_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
! CHECK: %[[P_DESC_LOAD:.*]] = fir.load %[[P_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[P_REBOX:.*]] = fir.rebox %[[P_DESC_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[P_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) (%[[P_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_REBOX:.*]] = fir.rebox %[[C3_LOAD]](%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: %[[PA_CONV:.*]] = fir.convert %[[PA_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C3_REBOX_CONV:.*]] = fir.convert %[[C3_REBOX]] : (!fir.class<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[PA_CONV]], %[[C3_REBOX_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD:.*]] = fir.load %[[PA_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_COORD:.*]] = fir.coordinate_of %[[PA_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[PA_EMBOX:.*]] = fir.embox %[[PA_COORD]] source_box %[[PA_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_REBOX:.*]] = fir.rebox %[[C4_LOAD]](%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: %[[PA_CONV:.*]] = fir.convert %[[PA_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>> 
! CHECK: %[[C4_REBOX_CONV:.*]] = fir.convert %[[C4_REBOX]] : (!fir.class<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.box<none> 
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[PA_CONV]], %[[C4_REBOX_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none 
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD:.*]] = fir.load %[[PA_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_COORD:.*]] = fir.coordinate_of %[[PA_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[PA_EMBOX:.*]] = fir.embox %[[PA_COORD]] source_box %[[PA_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[C4_DIMS:.*]]:3 = fir.box_dims %[[C4_LOAD]], %[[C0]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[C2:.*]] = arith.constant 2 : i64
! CHECK: %[[C2_INDEX:.*]] = fir.convert %[[C2]] : (i64) -> index
! CHECK: %[[C1:.*]] = arith.constant 1 : i64
! CHECK: %[[C1_INDEX:.*]] = fir.convert %[[C1]] : (i64) -> index
! CHECK: %[[C4:.*]] = arith.constant 4 : i64
! CHECK: %[[C4_INDEX:.*]] = fir.convert %[[C4]] : (i64) -> index
! CHECK: %[[SHIFT:.*]] = fir.shift %[[C4_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK: %[[SLICE:.*]] = fir.slice %[[C2_INDEX]], %[[C4_INDEX]], %[[C1_INDEX]] : (index, index, index) -> !fir.slice<1>
! CHECK: %[[SLICE_REBOX:.*]] = fir.rebox %[[C4_LOAD]](%[[SHIFT]]) [%[[SLICE]]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.class<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: %[[PA_CONV:.*]] = fir.convert %[[PA_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[SLICE_REBOX_CONV:.*]] = fir.convert %[[SLICE_REBOX]] : (!fir.class<!fir.array<3x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.box<none> 
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[PA_CONV]], %[[SLICE_REBOX_CONV]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none 
! CHECK-LABEL: fir.do_loop
! CHECK: %[[PA_LOAD:.*]] = fir.load %[[PA_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[PA_COORD:.*]] = fir.coordinate_of %[[PA_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[PA_EMBOX:.*]] = fir.embox %[[PA_COORD]] source_box %[[PA_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc"(%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[PA_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

end module

program test_pointer_association
  use poly
  call test_pointer()
end
