! Test that !$acc routine(name) with parallelism clauses for external
! subroutines correctly produces acc.routine with the keyword, even
! when the callee is a ProcEntity (not SubprogramDetails).

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-DAG: acc.routine @{{.*}} func(@_QPext_vec) vector
! CHECK-DAG: acc.routine @{{.*}} func(@_QPext_worker) worker
! CHECK-DAG: acc.routine @{{.*}} func(@_QPext_gang) gang
! CHECK-DAG: acc.routine @{{.*}} func(@_QPext_seq) seq

subroutine caller_vec(a, n)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: k
  !$acc routine(ext_vec) vector
  !$acc parallel loop gang
  do k = 1, 10
    call ext_vec(a, n)
  end do
end subroutine

subroutine caller_worker(a, n)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: k
  !$acc routine(ext_worker) worker
  !$acc parallel loop gang
  do k = 1, 10
    call ext_worker(a, n)
  end do
end subroutine

subroutine caller_gang(a, n)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  !$acc routine(ext_gang) gang
  !$acc parallel
  call ext_gang(a, n)
  !$acc end parallel
end subroutine

subroutine caller_seq(a, n)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: k
  !$acc routine(ext_seq) seq
  !$acc parallel loop gang vector
  do k = 1, 10
    call ext_seq(a, n)
  end do
end subroutine

! Test with explicit interface block (SubprogramDetails path).
! CHECK-DAG: acc.routine @{{.*}} func(@_QPext_iface) vector

subroutine caller_iface(a, n)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: k
  !$acc routine(ext_iface) vector
  interface
    subroutine ext_iface(a, n)
      integer, intent(in) :: n
      real, intent(inout) :: a(n)
    end subroutine
  end interface
  !$acc parallel loop gang
  do k = 1, 10
    call ext_iface(a, n)
  end do
end subroutine
