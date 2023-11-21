! Test generation of noinit, nofinal, and nodestroy fir.type_info attributes
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module tyinfo
  type boring_type
  end type

  type needs_final
    contains
      final :: needs_final_final
  end type

  type needs_init1
    integer :: i =0
  end type

  type needs_init_and_destroy
    integer, allocatable :: x
  end type

  type needs_all
    type(needs_final) :: x
    type(needs_init_and_destroy) :: y
  end type

  type, extends(needs_final) :: inherits_final
  end type
  type, extends(needs_init1) :: inherits_init
  end type
  type, extends(needs_init_and_destroy) :: inherits_init_and_destroy
  end type
  type, extends(needs_all) :: inherits_all
  end type

  interface
    subroutine needs_final_final(x)
      type(needs_final), intent(inout) :: x
    end subroutine
  end interface

  type(boring_type) :: x1
  type(needs_final) :: x2
  type(needs_init1) :: x3
  type(needs_init_and_destroy) :: x4
  type(needs_all) :: x5
  type(inherits_final) :: x6
  type(inherits_init) :: x7
  type(inherits_init_and_destroy) :: x8
  type(inherits_all) :: x9
end module

! CHECK-DAG:  fir.type_info @_QMtyinfoTboring_type noinit nodestroy nofinal : !fir.type<_QMtyinfoTboring_type>
! CHECK-DAG:  fir.type_info @_QMtyinfoTneeds_final noinit : !fir.type<_QMtyinfoTneeds_final>
! CHECK-DAG:  fir.type_info @_QMtyinfoTneeds_init1 nodestroy nofinal : !fir.type<_QMtyinfoTneeds_init1{i:i32}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTneeds_init_and_destroy nofinal : !fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTneeds_all : !fir.type<_QMtyinfoTneeds_all{x:!fir.type<_QMtyinfoTneeds_final>,y:!fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}>}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTinherits_final noinit extends !fir.type<_QMtyinfoTneeds_final> : !fir.type<_QMtyinfoTinherits_final{needs_final:!fir.type<_QMtyinfoTneeds_final>}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTinherits_init nodestroy nofinal extends !fir.type<_QMtyinfoTneeds_init1{i:i32}> : !fir.type<_QMtyinfoTinherits_init{needs_init1:!fir.type<_QMtyinfoTneeds_init1{i:i32}>}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTinherits_init_and_destroy nofinal extends !fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}> : !fir.type<_QMtyinfoTinherits_init_and_destroy{needs_init_and_destroy:!fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}>}>
! CHECK-DAG:  fir.type_info @_QMtyinfoTinherits_all extends !fir.type<_QMtyinfoTneeds_all{x:!fir.type<_QMtyinfoTneeds_final>,y:!fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}>}> : !fir.type<_QMtyinfoTinherits_all{needs_all:!fir.type<_QMtyinfoTneeds_all{x:!fir.type<_QMtyinfoTneeds_final>,y:!fir.type<_QMtyinfoTneeds_init_and_destroy{x:!fir.box<!fir.heap<i32>>}>}>}>
