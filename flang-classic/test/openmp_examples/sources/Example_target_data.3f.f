! @@name:	target_data.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine gramSchmidt(Q,rows,cols)
integer             ::   rows,cols,  i,k
double precision    :: Q(rows,cols), tmp
      !$omp target data map(Q)
      do k=1,cols
         tmp = 0.0d0
        !$omp target
           !$omp parallel do reduction(+:tmp)
           do i=1,rows
              tmp = tmp + (Q(i,k) * Q(i,k))
           end do
        !$omp end target
          tmp = 1.0d0/sqrt(tmp)
        !$omp target
           !$omp parallel do
           do i=1,rows
               Q(i,k) = Q(i,k)*tmp
           enddo
        !$omp end target
      end do
      !$omp end target data
end subroutine
