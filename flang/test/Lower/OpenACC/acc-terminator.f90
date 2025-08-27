! Check that acc.terminator is not inserted in data construct

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

program main
  use, intrinsic :: iso_c_binding
  implicit none

  real(8), pointer :: a(:,:,:),b(:,:,:),c(:,:,:),c2(:,:,:)
  integer, parameter :: n1 = 400, n2 = 20
  integer*4 :: stat
  integer :: i,j,k

  stat = 0
  do i=1,n2

    !$acc data copyin(a(:,:,i),b(:,:,i),c(:,:,i)) copyout(c2(:,:,i))

    !$acc host_data use_device(a(:,:,i),b(:,:,i),c(:,:,i))
    
    !$acc end host_data

    if ( stat .ne. 0 ) then
      print *, "stat = ",stat
      stop ! terminator here should be fir.unreachable
    end if

    !$acc parallel loop present(c(:,:,i),c2(:,:,i))
    do j = 1,n1
       do k = 1,n1
          c2(k,j,i) = 1.5d0 * c(k,j,i)
       enddo
    enddo
    !$acc end parallel loop

    !$acc end data

  enddo

  !$acc wait

  deallocate(a,b,c,c2)
end program

! CHECK-LABEL: func.func @_QQmain()
! CHECK: acc.data
! CHECK: acc.host_data
! CHECK: acc.terminator
! CHECK: fir.call @_FortranAStopStatement
! CHECK: fir.unreachable
! CHECK: acc.parallel
! CHECK-COUNT-3: acc.yield
! CHECK: acc.terminator
