! This test checks lowering of OpenMP declare reduction Directive, with initialization
! via a subroutine. This functionality is currently not implemented.

! RUN: not flang -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: OpenMPDeclareReductionConstruct
subroutine initme(x,n)
  integer x,n
  x=n
end subroutine initme

function func(x, n, init)
  integer func
  integer x(n)
  integer res
  interface
     subroutine initme(x,n)
       integer x,n
     end subroutine initme
  end interface
!$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,0))
  res=init
!$omp simd reduction(red_add:res)
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func
