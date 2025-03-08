! Test that temps are always created of parenthesized arguments in
! calls.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfoo_num_scalar(
subroutine foo_num_scalar(x)
  integer :: x
  call bar_num_scalar(x)
! CHECK-NOT: fir.load
! CHECK:  fir.call @_QPbar_num_scalar(
  call bar_num_scalar((x))
! CHECK:  fir.load
! CHECK:  fir.call @_QPbar_num_scalar(
end subroutine

! CHECK-LABEL: func @_QPfoo_char_scalar(
subroutine foo_char_scalar(x)
  character(5) :: x
! CHECK-NOT:  hlfir.as_expr
! CHECK:  fir.call @_QPbar_char_scalar(
  call bar_char_scalar(x)
! CHECK:  hlfir.as_expr
! CHECK:  fir.call @_QPbar_char_scalar(
  call bar_char_scalar((x))
end subroutine

! CHECK-LABEL: func @_QPfoo_num_array(
subroutine foo_num_array(x)
  integer :: x(100)
  call bar_num_array(x)
! CHECK-NOT:  hlfir.elemental
! CHECK:  fir.call @_QPbar_num_array(
  call bar_num_array((x))
! CHECK:  hlfir.elemental
! CHECK:  fir.call @_QPbar_num_array(
end subroutine

! CHECK-LABEL: func @_QPfoo_char_array(
subroutine foo_char_array(x)
  character(10) :: x(100)
  call bar_char_array(x)
! CHECK-NOT:  hlfir.elemental
! CHECK:  fir.call @_QPbar_char_array(
  call bar_char_array((x))
! CHECK:  hlfir.elemental
! CHECK:  fir.call @_QPbar_char_array(
end subroutine

! CHECK-LABEL: func @_QPfoo_num_array_box(
subroutine foo_num_array_box(x)
  integer :: x(100)
  interface
   subroutine bar_num_array_box(x)
     integer :: x(:)
   end subroutine
  end interface
  call bar_num_array_box(x)
! CHECK-NOT:  hlfir.elemental
! CHECK:  fir.call @_QPbar_num_array_box(
  call bar_num_array_box((x))
! CHECK:  hlfir.elemental
! CHECK:  fir.call @_QPbar_num_array_box(
end subroutine

! CHECK-LABEL: func @_QPfoo_char_array_box(
subroutine foo_char_array_box(x, n)

  integer :: n
  character(10) :: x(n)
  interface
   subroutine bar_char_array_box(x)
     character(*) :: x(:)
   end subroutine
  end interface
  call bar_char_array_box(x)
! CHECK-NOT:  hlfir.elemental
! CHECK:  fir.call @_QPbar_char_array_box(
  call bar_char_array_box((x))
! CHECK:  hlfir.elemental
! CHECK:  fir.call @_QPbar_char_array_box(
end subroutine
