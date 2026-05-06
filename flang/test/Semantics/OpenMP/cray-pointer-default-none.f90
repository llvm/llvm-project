!RUN: %python %S/../test_errors.py %s %flang -fopenmp

! None of the below tests should fail.

subroutine none_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_shared', var(1)
  !$omp end parallel
end subroutine

subroutine none_private()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) private(ivar) shared(pointee)
    ivar = loc(pointee)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_private', var(1)
  !$omp end parallel
end subroutine

subroutine none_firstprivate()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) firstprivate(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_firstprivate', var(1)
  !$omp end parallel
end subroutine

subroutine private_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(private) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'private_shared', var(1)
  !$omp end parallel
end subroutine

subroutine private_firstprivate()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(private) firstprivate(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'private_firstprivate', var(1)
  !$omp end parallel
end subroutine

subroutine firstprivate_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(firstprivate) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'firstprivate_shared', var(1)
  !$omp end parallel
end subroutine

subroutine firstprivate_private()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(firstprivate) private(ivar) shared(pointee)
    ivar = loc(pointee)
    var(1) = var(1) / 2
    print '(A24,I6)', 'firstprivate_private', var(1)
  !$omp end parallel
end subroutine

subroutine loop_common_none_shared()
  implicit none
  common /cmn/ mp
  real a(1)
  pointer(mp,a)
  integer i

  !$omp parallel num_threads(1) default(none) shared(mp)
    !$omp do
    do i = 1,10
        a(1) = a(1) / 2
    end do
    print '(A24,I6)', 'none_shared', a(1)
  !$omp end parallel
end subroutine

subroutine loop_common_firstprivate()
  implicit none
  common /cmn/ ptr
  real a(1)
  pointer(ptr,a)
  integer i

  !$omp parallel num_threads(1) default(firstprivate) shared(ptr)
    !$omp do
    do i = 1,10
        a(1) = a(1) / 2
    end do
    print '(A24,I6)', 'none_shared', a(1)
  !$omp end parallel
end subroutine

subroutine loop_common_private()
  implicit none
  common /cmn/ ptr
  real a(1)
  pointer(ptr,a)
  integer i

  !$omp parallel num_threads(1) default(firstprivate) private(ptr)
    !$omp do
    do i = 1,10
        a(1) = a(1) / 2
    end do
    print '(A24,I6)', 'none_shared', a(1)
  !$omp end parallel
end subroutine
