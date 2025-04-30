! @@name:	task_dep.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
! Assume BS divides N perfectly
subroutine matmul_depend (N, BS, A, B, C)
   implicit none
   integer :: N, BS, BM
   real, dimension(N, N) :: A, B, C
   integer :: i, j, k, ii, jj, kk
   BM = BS - 1
   do i = 1, N, BS
      do j = 1, N, BS
         do k = 1, N, BS
!$omp task shared(A,B,C) private(ii,jj,kk) & ! I,J,K are firstprivate by default
!$omp depend ( in: A(i:i+BM, k:k+BM), B(k:k+BM, j:j+BM) ) &
!$omp depend ( inout: C(i:i+BM, j:j+BM) )
            do ii = i, i+BM
               do jj = j, j+BM
                  do kk = k, k+BM
                     C(jj,ii) = C(jj,ii) + A(kk,ii) * B(jj,kk)
                  end do
               end do
            end do
!$omp end task
         end do
      end do
   end do
end subroutine
