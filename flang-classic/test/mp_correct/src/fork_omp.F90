program forking
  implicit none
  integer :: pid

  pid = -1
  call sub
  call fork(pid)
  if (pid /= 0) then
    call waitpid(pid)
  endif
  call sub
  if (pid /= 0) then
#ifdef _OPENMP
    call check(1, 1, 1)
#else
    call check(0, 1, 1)
#endif
  endif

contains

  subroutine sub
    use omp_lib
    implicit none
!$omp parallel
!$omp critical
    print *, omp_get_thread_num()
!$omp end critical
!$omp end parallel
  end subroutine
end program
