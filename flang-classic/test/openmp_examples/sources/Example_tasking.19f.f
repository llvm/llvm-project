! @@name:	tasking.19f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine matmul_depend (N, BS, A, B, C)
   integer :: N, BS, BM
   real, dimension(N, N) :: A, B, C
   integer :: i, j, k, ii, jj, kk
   BM = BS -1
   do i = 1, N, BS
      do j = 1, N, BS
         do k = 1, N, BS
!$omp task depend ( in: A(i:i+BM, k:k+BM), B(k:k+BM, j:j+BM) ) &
!$omp depend ( inout: C(i:i+BM, j:j+BM) )
            do ii = i, i+BS
               do jj = j, j+BS
                  do kk = k, k+BS
                     C(jj,ii) = C(jj,ii) + A(kk,ii) * B(jj,kk)
                  end do
               end do
            end do
!$omp end task
         end do
      end do
   end do
end subroutine
