! @@name:	array_sections.2f
! @@type:	F-free
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
subroutine foo()
integer,target  :: A(30)
integer,pointer :: p(:)
   A=1
   !$omp target data map( A(1:4) )
     p=>A
     ! invalid because p(4) and A(4) are the same
     ! location on the host but the array section
     ! specified via p(...) is not a subset of A(1:4)
     !$omp target map( p(4:23) )   ! { error "Insert compiler error message text here" }
        A(3) = 0
        p(9) = 0
     !$omp end target
   !$omp end target data
end subroutine
