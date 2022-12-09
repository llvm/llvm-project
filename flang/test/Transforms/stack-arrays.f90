! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --array-value-copy | fir-opt --stack-arrays | FileCheck %s

! check simple array value copy case
subroutine array_value_copy_simple(arr)
  integer, intent(inout) :: arr(4)
  arr(3:4) = arr(1:2)
end subroutine
! CHECK-LABEL: func.func @_QParray_value_copy_simple(%arg0: !fir.ref<!fir.array<4xi32>>
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: fir.alloca !fir.array<4xi32>
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }

! check complex array value copy case
module stuff
  type DerivedWithAllocatable
    integer, dimension(:), allocatable :: dat
  end type

  contains
  subroutine array_value_copy_complex(arr)
    type(DerivedWithAllocatable), intent(inout) :: arr(:)
    arr(3:4) = arr(1:2)
  end subroutine
end module
! CHECK: func.func
! CHECK-SAME: array_value_copy_complex
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: fir.alloca !fir.array<?x!fir.type<_QMstuffTderivedwithallocatable
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }

subroutine parameter_array_init
  integer, parameter :: p(100) = 42
  call use_p(p)
end subroutine
! CHECK: func.func
! CHECK-SAME: parameter_array_init
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: fir.alloca !fir.array<100xi32>
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }

subroutine test_vector_subscripted_section_to_box(v, x)
  interface
    subroutine takes_box(y)
      real :: y(:)
    end subroutine
  end interface

  integer :: v(:)
  real :: x(:)
  call takes_box(x(v))
end subroutine
! CHECK: func.func
! CHECK-SAME: test_vector_subscripted_section_to_box
! CHECK-NOT: fir.allocmem
! CHECK: fir.alloca !fir.array<?xf32>
! CHECK-NOT: fir.allocmem
! CHECK: fir.call @_QPtakes_box
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }

subroutine call_parenthesized_arg(x)
  integer :: x(100)
  call bar((x))
end subroutine
! CHECK: func.func
! CHECK-SAME: call_parenthesized_arg
! CHECK-NOT: fir.allocmem
! CHECK: fir.alloca !fir.array<100xi32>
! CHECK-NOT: fir.allocmem
! CHECK: fir.call @_QPbar
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }

subroutine where_allocatable_assignments(a, b)
  integer :: a(:)
  integer, allocatable :: b(:)
  where(b > 0)
    b = a
  elsewhere
    b(:) = 0
  end where
end subroutine
! TODO: broken: passing allocation through fir.result
! CHECK: func.func
! CHECK-SAME: where_allocatable_assignments
! CHECK: return
! CHECK-NEXT: }

subroutine array_constructor(a, b)
  real :: a(5), b
  real, external :: f
  a = [f(b), f(b+1), f(b+2), f(b+5), f(b+11)]
end subroutine
! TODO: broken: realloc
! CHECK: func.func
! CHECK-SAME: array_constructor
! CHECK: return
! CHECK-NEXT: }

subroutine sequence(seq, n)
  integer :: n, seq(n)
  seq = [(i,i=1,n)]
end subroutine
! TODO: broken: realloc
! CHECK: func.func
! CHECK-SAME: sequence
! CHECK: return
! CHECK-NEXT: }

subroutine CFGLoop(x)
  integer, parameter :: k = 100, m=1000000, n = k*m
  integer :: x(n)
  logical :: has_error

  do i=0,m-1
    x(k*i+1:k*(i+1)) = x(k*(i+1):k*i+1:-1)
    if (has_error(x, k)) stop
  end do
end subroutine
! CHECK: func.func
! CHECK-SAME: cfgloop
! CHECK-NEXT: %[[MEM:.*]] = fir.alloca !fir.array<100000000xi32>
! CHECK-NOT: fir.allocmem
! CHECK-NOT: fir.freemem
! CHECK: return
! CHECK-NEXT: }
