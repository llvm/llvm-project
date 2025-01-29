! Test generation of fir.dt_component
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
subroutine test_1(x)
  integer, save, target :: my_target
  procedure() :: my_proc
  type :: sometype
    integer :: i(-1:8) = 42
    integer :: j(1:8)
    integer, allocatable :: alloc(:)
    integer, pointer :: p => my_target
    integer, pointer :: p2 => NULL()
    integer, pointer :: p3
    procedure(), pointer, nopass :: proc_p => my_proc
    procedure(), pointer, nopass :: proc_p2 => NULL()
    procedure(), pointer, nopass :: proc_p3
  end type
  type(sometype) :: x
end subroutine
! CHECK-LABEL:   fir.type_info @_QFtest_1Tsometype
! CHECK-SAME     component_info {
! CHECK:           fir.dt_component "i" lbs [-1] init @_QFtest_1E.di.sometype.i
! CHECK-NOT:       fir.dt_component "j"
! CHECK:           fir.dt_component "p" init @_QFtest_1E.di.sometype.p
! CHECK:           fir.dt_component "p2" init @_QFtest_1E.di.sometype.p2
! CHECK:           fir.dt_component "proc_p" init @_QPmy_proc
! CHECK:         }

subroutine test_nesting(x)
  type some_sub_type
    integer :: i = 42
  end type
  type sometype2
    type(some_sub_type) :: nested
  end type
  type(sometype2) :: x
end subroutine
! CHECK-LABEL:   fir.type_info @_QFtest_nestingTsome_sub_type
! CHECK-SAME     component_info {
! CHECK:           fir.dt_component "i" init @_QFtest_nestingE.di.some_sub_type.i
! CHECK:         }

! CHECK:         fir.type_info @_QFtest_nestingTsometype2
! CHECK-NOT:       fir.dt_component


subroutine data_like(x)
  type sometype3
    integer :: i/42/
  end type
  type(sometype3) :: x
end subroutine
! CHECK-LABEL:   fir.type_info @_QFdata_likeTsometype3
! CHECK-SAME     component_info {
! CHECK:           fir.dt_component "i" init @_QFdata_likeE.di.sometype3.i
! CHECK:         }
