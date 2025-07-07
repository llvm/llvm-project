! RUN: not %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s

module mm
  implicit none
  type two
     integer(4) :: a, b
  end type two

  type three
     integer(8) :: a, b, c
  end type three
contains
  function add_two(x, y)
    type(two) add_two, x, y, res
    add_two = res
  end function add_two

  function func(n)
    type(three) :: func
    type(three) :: res3
    integer :: n
    integer :: i

    !$omp declare reduction(dummy:kerflunk:omp_out=omp_out+omp_in)
!CHECK: error: Derived type 'kerflunk' not found
    
    !$omp declare reduction(adder:two:omp_out=add_two(omp_out,omp_in))
    !$omp simd reduction(adder:res3)
!CHECK: error: The type of 'res3' is incompatible with the reduction operator.
    do i=1,n
    enddo
    func = res3
  end function func
end module mm
