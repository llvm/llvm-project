! Test lowering of elemental subroutine calls with array arguments
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_elem_sub(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<complex<f32>>{{.*}}) {
! CHECK:         %[[c_decl:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:         %[[i_decl:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:         %[[x_decl:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:         %[[z_decl:.*]]:2 = hlfir.declare %[[VAL_3]]
! CHECK:         %[[len:.*]] = fir.box_elesize %[[c_decl]]#1
! CHECK:         %[[slice:.*]] = hlfir.designate %[[c_decl]]#0 ({{.*}})  shape {{.*}} typeparams %[[len]]
! CHECK:         %[[z_val:.*]] = fir.load %[[z_decl]]#0
! CHECK:         fir.do_loop %[[arg:.*]] = {{.*}} {
! CHECK:           %[[x_elem:.*]] = hlfir.designate %[[x_decl]]#0 (%[[arg]])
! CHECK:           %[[c_elem:.*]] = hlfir.designate %[[slice]] (%[[arg]]) typeparams %[[len]]
! CHECK:           fir.call @_QPfoo(%[[x_elem]], %[[c_elem]], %[[i_decl]]#0, %[[z_val]])
! CHECK:         }

subroutine test_elem_sub(x, c, i, z)
  real :: x(:)
  character(*) :: c(:)
  integer :: i
  complex :: z
  interface
    elemental subroutine foo(x, c, i, z)
      real, intent(out) :: x
      character(*), intent(inout) :: c
      integer, intent(in) :: i
      complex, value :: z
    end subroutine
  end interface

  call foo(x, c(10:1:-1), i, z)
end subroutine

! CHECK-LABEL: func @_QPtest_elem_sub_no_array_args(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_elem_sub_no_array_args(i, j)
  integer :: i, j
  interface
    elemental subroutine bar(i, j)
      integer, intent(out) :: i
      integer, intent(in) :: j
    end subroutine
  end interface
  call bar(i, j)
  ! CHECK:         fir.call @_QPbar(%{{.*}}, %{{.*}})
end subroutine
