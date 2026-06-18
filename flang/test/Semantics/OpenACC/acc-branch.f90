! RUN: %python %S/../test_errors.py %s %flang -fopenacc -pedantic

! Check OpenACC restruction in branch in and out of some construct
!
subroutine openacc_clause_validity

  implicit none

  integer :: i, j, k
  integer :: N = 256
  real(8) :: a(256)

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a PARALLEL construct
    return
  end do
  !$acc end parallel

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a PARALLEL LOOP construct
    return
  end do

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a SERIAL LOOP construct
    return
  end do

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a KERNELS LOOP construct
    return
  end do

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      exit
    end if
  end do
  !$acc end parallel

  ! Exit branches out of parallel construct, not attached to an OpenACC parallel construct.
  name1: do k=1, N
  !$acc parallel
  !$acc loop
  outer: do i=1, N
    inner: do j=1, N
      ifname: if (j == 2) then
        ! These are allowed.
        exit
        exit inner
        exit outer
        !ERROR: EXIT to construct 'name1' outside of PARALLEL construct is not allowed
        exit name1
        ! Exit to construct other than loops.
        exit ifname
      end if ifname
    end do inner
  end do outer
  !$acc end parallel
  end do name1

  ! Exit branches out of parallel construct, attached to an OpenACC parallel construct.
  thisblk: BLOCK
    fortname: if (.true.) then
      !PORTABILITY: The construct name 'name1' should be distinct at the subprogram level [-Wbenign-name-clash]
      name1: do k = 1, N
        !$acc parallel
        !ERROR: EXIT to construct 'fortname' outside of PARALLEL construct is not allowed
        exit fortname
        !$acc loop
          do i = 1, N
            a(i) = 3.14d0
            if(i == N-1) THEN
              !ERROR: EXIT to construct 'name1' outside of PARALLEL construct is not allowed
              exit name1
            end if
          end do

          loop2: do i = 1, N
            a(i) = 3.33d0
            !ERROR: EXIT to construct 'thisblk' outside of PARALLEL construct is not allowed
            exit thisblk
          end do loop2
        !$acc end parallel
      end do name1
    end if fortname
  end BLOCK thisblk

  !Exit branches inside OpenACC construct.
  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
    ifname: if (i == 2) then
      ! This is allowed.
      exit ifname
    end if ifname
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      stop 999 ! no error
    end if
  end do
  !$acc end parallel

  !$acc kernels
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a KERNELS construct
    return
  end do
  !$acc end kernels

  !$acc kernels
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      exit
    end if
  end do
  !$acc end kernels

  !$acc kernels
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      stop 999 ! no error
    end if
  end do
  !$acc end kernels

  !$acc serial
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: RETURN statement is not allowed in a SERIAL construct
    return
  end do
  !$acc end serial

  !$acc serial
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      exit
    end if
  end do
  !$acc end serial

  name2: do k=1, N
  !$acc serial
  do i = 1, N
    ifname: if (.true.) then
      print *, "LGTM"
    a(i) = 3.14d0
    if(i == N-1) THEN
        !ERROR: EXIT to construct 'name2' outside of SERIAL construct is not allowed
        exit name2
        exit ifname
      end if
    end if ifname
    end do
  !$acc end serial
  end do name2

  !$acc serial
  do i = 1, N
    a(i) = 3.14d0
    if(i == N-1) THEN
      stop 999 ! no error
    end if
  end do
  !$acc end serial


  !$acc data create(a)

  !ERROR: RETURN statement is not allowed in a DATA construct
  if (size(a) == 10) return

  !$acc end data

  ! GOTO branching out of compute constructs is not allowed (spec 2.5.4).
  !$acc parallel
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: GOTO to a label outside of a PARALLEL construct is not allowed
    if (i == N-1) goto 999
  end do
  !$acc end parallel
999 continue

  !$acc kernels
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: GOTO to a label outside of a KERNELS construct is not allowed
    if (i == N-1) goto 998
  end do
  !$acc end kernels
998 continue

  !$acc serial
  do i = 1, N
    a(i) = 3.14d0
    !ERROR: GOTO to a label outside of a SERIAL construct is not allowed
    if (i == N-1) goto 997
  end do
  !$acc end serial
997 continue

  ! GOTO within a compute construct is allowed.
  !$acc parallel
  do i = 1, N
    if (i == N-1) goto 996
996 a(i) = 3.14d0
  end do
  !$acc end parallel

  ! GOTO out of a data construct is allowed.
  !$acc data create(a)
  do i = 1, N
    if (i == N-1) goto 995
  end do
  !$acc end data
995 continue

end subroutine openacc_clause_validity
