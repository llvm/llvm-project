! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests the different possible type involving polymorphic entities.

module call_dispatch

  interface
    subroutine nopass_defferred(x)
      real :: x(:)
    end subroutine
  end interface

  type p1
    integer :: a
    integer :: b
  contains
    procedure, nopass :: tbp_nopass
    procedure :: tbp_pass
    procedure, pass(this) :: tbp_pass_arg0
    procedure, pass(this) :: tbp_pass_arg1

    procedure, nopass :: proc1 => p1_proc1_nopass
    procedure :: proc2 => p1_proc2
    procedure, pass(this) :: proc3 => p1_proc3_arg0
    procedure, pass(this) :: proc4 => p1_proc4_arg1

    procedure, nopass :: p1_fct1_nopass
    procedure :: p1_fct2
    procedure, pass(this) :: p1_fct3_arg0
    procedure, pass(this) :: p1_fct4_arg1

    procedure :: pass_with_class_arg
  end type

  type, abstract :: a1
    real :: a
    real :: b
  contains
    procedure(nopass_defferred), deferred, nopass :: nopassd
  end type

  type :: node
    type(node_ptr), pointer :: n(:)
  end type
  type :: use_node
    type(node) :: n
  end type
  type :: node_ptr
    type(node_ptr), pointer :: n
  end type

  contains

! ------------------------------------------------------------------------------
! Test lowering of type-bound procedure call on polymorphic entities
! ------------------------------------------------------------------------------

    function p1_fct1_nopass()
      real :: p1_fct1_nopass
    end function
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_fct1_nopass() -> f32

    function p1_fct2(p)
      real :: p1_fct2
      class(p1) :: p
    end function
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_fct2(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32

    function p1_fct3_arg0(this)
      real :: p1_fct2
      class(p1) :: this
    end function
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_fct3_arg0(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32

    function p1_fct4_arg1(i, this)
      real :: p1_fct2
      integer :: i
      class(p1) :: this
    end function
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_fct4_arg1(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32

    subroutine pass_with_class_arg(this, other)
      class(p1) :: this
      class(p1) :: other
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPpass_with_class_arg(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, %{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {

    subroutine p1_proc1_nopass()
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_proc1_nopass()

    subroutine p1_proc2(p)
      class(p1) :: p
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_proc2(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine p1_proc3_arg0(this)
      class(p1) :: this
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_proc3_arg0(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine p1_proc4_arg1(i, this)
      integer, intent(in) :: i
      class(p1) :: this
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPp1_proc4_arg1(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine tbp_nopass()
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPtbp_nopass()

    subroutine tbp_pass(t)
      class(p1) :: t
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPtbp_pass(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine tbp_pass_arg0(this)
      class(p1) :: this
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPtbp_pass_arg0(%{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine tbp_pass_arg1(i, this)
      integer, intent(in) :: i
      class(p1) :: this
    end subroutine
    ! CHECK-LABEL: func.func @_QMcall_dispatchPtbp_pass_arg1(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)

    subroutine check_dispatch(p)
      class(p1) :: p
      real :: a

      call p%tbp_nopass()
      call p%tbp_pass()
      call p%tbp_pass_arg0()
      call p%tbp_pass_arg1(1)

      call p%proc1()
      call p%proc2()
      call p%proc3()
      call p%proc4(1)

      a = p%p1_fct1_nopass()
      a = p%p1_fct2()
      a = p%p1_fct3_arg0()
      a = p%p1_fct4_arg1(1)
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch(
! CHECK-SAME:  %[[P:.*]]: !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>> {fir.bindc_name = "p"}) {
! CHECK:       fir.dispatch "tbp_nopass"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>){{$}}
! CHECK:       fir.dispatch "tbp_pass"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK:       fir.dispatch "tbp_pass_arg0"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK:       fir.dispatch "tbp_pass_arg1"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>)  (%{{.*}}, %[[P]] : !fir.ref<i32>, !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32} 

! CHECK:       fir.dispatch "proc1"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>){{$}}
! CHECK:       fir.dispatch "proc2"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK:       fir.dispatch "proc3"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK:       fir.dispatch "proc4"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%{{.*}}, %[[P]] : !fir.ref<i32>, !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}

! CHECK: %{{.*}} = fir.dispatch "p1_fct1_nopass"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32{{$}}
! CHECK: %{{.*}} = fir.dispatch "p1_fct2"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32 {pass_arg_pos = 0 : i32}
! CHECK: %{{.*}} = fir.dispatch "p1_fct3_arg0"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32 {pass_arg_pos = 0 : i32}
! CHECK: %{{.*}} = fir.dispatch "p1_fct4_arg1"(%[[P]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%{{.*}}, %[[P]] : !fir.ref<i32>, !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> f32 {pass_arg_pos = 1 : i32}

  subroutine check_dispatch_deferred(a, x)
    class(a1) :: a
    real :: x(:)
    call a%nopassd(x)
  end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_deferred(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMcall_dispatchTa1{a:f32,b:f32}>> {fir.bindc_name = "a"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK: fir.dispatch "nopassd"(%[[ARG0]] : !fir.class<!fir.type<_QMcall_dispatchTa1{a:f32,b:f32}>>) (%[[ARG1]] : !fir.box<!fir.array<?xf32>>)

    subroutine check_dispatch_scalar_allocatable(p)
      class(p1), allocatable :: p
      call p%tbp_pass()
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_scalar_allocatable(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>> {fir.bindc_name = "p"}) {
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[REBOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%1 : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

    subroutine check_dispatch_scalar_pointer(p)
      class(p1), pointer :: p
      call p%tbp_pass()
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_scalar_pointer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>> {fir.bindc_name = "p"}) {
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[REBOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%1 : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

    subroutine check_dispatch_static_array(p, t)
      class(p1) :: p(10)
      type(p1) :: t(10)
      integer :: i
      do i = 1, 10
        call p(i)%tbp_pass()
      end do

      do i = 1, 10
        call t(i)%tbp_pass()
      end do
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_static_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<10x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "t"}) {
! CHECK: fir.do_loop {{.*}} {
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.class<!fir.array<10x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS_BOX:.*]] = fir.embox %[[COORD]] source_box %[[ARG0]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.array<10x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: fir.do_loop {{.*}} {
! CHECK: %[[COORD1:.*]] = fir.coordinate_of %[[ARG1]], %{{.*}} : (!fir.ref<!fir.array<10x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMcall_dispatchPtbp_pass(%[[EMBOX]]) {{.*}}: (!fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> ()

    subroutine check_dispatch_dynamic_array(p, t)
      class(p1) :: p(:)
      type(p1) :: t(:)
      integer :: i
      do i = 1, 10
        call p(i)%tbp_pass()
      end do

      do i = 1, 10
        call t(i)%tbp_pass()
      end do
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_dynamic_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "t"}) {
! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS_BOX:.*]] = fir.embox %[[COORD]] source_box %[[ARG0]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG1]], %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMcall_dispatchPtbp_pass(%[[EMBOX]]) {{.*}}: (!fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> ()

    subroutine check_dispatch_allocatable_array(p, t)
      class(p1), allocatable :: p(:)
      type(p1), allocatable :: t(:)
      integer :: i
      do i = 1, 10
        call p(i)%tbp_pass()
      end do

      do i = 1, 10
        call t(i)%tbp_pass()
      end do
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_allocatable_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "t"}) {
! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: fir.store %arg3 to %0 : !fir.ref<i32>
! CHECK: %[[LOAD_ARG0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS_ARG0:.*]]:3 = fir.box_dims %[[LOAD_ARG0]], %[[C0]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[LOAD_ARG0]], %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS_BOX:.*]] = fir.embox %[[COORD]] source_box %[[LOAD_ARG0]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[LOAD_ARG1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS_ARG1:.*]]:3 = fir.box_dims %[[LOAD_ARG1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[LOAD_ARG1]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMcall_dispatchPtbp_pass(%[[EMBOX]]) {{.*}}: (!fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> ()

    subroutine check_dispatch_pointer_array(p, t)
      class(p1), pointer :: p(:)
      type(p1), pointer :: t(:)
      integer :: i
      do i = 1, 10
        call p(i)%tbp_pass()
      end do

      do i = 1, 10
        call t(i)%tbp_pass()
      end do
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "t"}) {

! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[LOAD_ARG0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS_ARG0]]:3 = fir.box_dims %[[LOAD_ARG0]], %[[C0]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[LOAD_ARG0]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS_BOX]] = fir.embox %[[COORD]] source_box %[[LOAD_ARG0]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "tbp_pass"(%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[CLASS_BOX]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[LOAD_ARG1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS_ARG1:.*]]:3 = fir.box_dims %[[LOAD_ARG1]], %[[C0]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[LOAD_ARG1]], %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMcall_dispatchPtbp_pass(%[[EMBOX]]) {{.*}}: (!fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) -> ()

    subroutine check_dispatch_dynamic_array_copy(p, o)
      class(p1) :: p(:)
      class(p1) :: o(:)

      integer :: i
      do i = 1, 9
        call p(i)%pass_with_class_arg(o(i+1))
      end do
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_dispatch_dynamic_array_copy(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}, 
! CHECK-SAME: %[[ARG1:.*]]: !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>> {fir.bindc_name = "o"}) {
! CHECK: %{{.*}} = fir.do_loop {{.*}} {
! CHECK: %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS1:.*]] = fir.embox %[[COORD1]] source_box %[[ARG0]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: %[[CLASS2:.*]] = fir.embox %[[COORD2]] source_box %[[ARG1]] : (!fir.ref<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "pass_with_class_arg"(%[[CLASS1]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) (%[[CLASS1]], %[[CLASS2]] : !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>, !fir.class<!fir.type<_QMcall_dispatchTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! ------------------------------------------------------------------------------
! Test that direct call is emitted when the type is known
! ------------------------------------------------------------------------------

    subroutine check_nodispatch(t)
      type(p1) :: t
      call t%tbp_nopass()
      call t%tbp_pass()
      call t%tbp_pass_arg0()
      call t%tbp_pass_arg1(1)
    end subroutine

! CHECK-LABEL: func.func @_QMcall_dispatchPcheck_nodispatch
! CHECK: fir.call @_QMcall_dispatchPtbp_nopass
! CHECK: fir.call @_QMcall_dispatchPtbp_pass
! CHECK: fir.call @_QMcall_dispatchPtbp_pass_arg0
! CHECK: fir.call @_QMcall_dispatchPtbp_pass_arg1

  subroutine use_node_test(n)
    type(use_node) :: n
  end subroutine

end module
