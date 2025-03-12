! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

!CHECK-LABEL: Subprogram scope: initme
subroutine initme(x,n)
  integer x,n
  x=n
end subroutine initme

!CHECK-LABEL: Subprogram scope: func
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
!CHECK: red_add: Misc ConstructName
!CHECK: Subprogram scope: initme  
!$omp simd reduction(red_add:res)
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func

program main
!CHECK-LABEL: MainProgram scope: main

  !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)

!CHECK: my_add_red: Misc ConstructName
  
end program main

