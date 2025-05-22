! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! 2.15.1.1 Predetermined rules for associated do-loops index variable
!   a) The loop iteration variable(s) in the associated do-loop(s) of a do,
!      parallel do, taskloop, or distribute construct is (are) private.
!   b) The loop iteration variable in the associated do-loop of a simd construct
!      with just one associated do-loop is linear with a linear-step that is the
!      increment of the associated do-loop.
!   c) The loop iteration variables in the associated do-loops of a simd
!      construct with multiple associated do-loops are lastprivate.
!   d) A loop iteration variable for a sequential loop in a parallel or task
!      generating construct is private in the innermost such construct that
!      encloses the loop.
!   - TBD

! All the tests assume that the do-loops association for collapse/ordered
! clause has been performed (the number of nested do-loops >= n).

! Rule a)
! TODO: nested constructs (k should be private too)
!DEF: /test_do (Subroutine) Subprogram
subroutine test_do
 implicit none
 !DEF: /test_do/a ObjectEntity REAL(4)
 real a(20,20,20)
 !DEF: /test_do/i ObjectEntity INTEGER(4)
 !DEF: /test_do/j ObjectEntity INTEGER(4)
 !DEF: /test_do/k ObjectEntity INTEGER(4)
 integer i, j, k
!$omp parallel
 !DEF: /test_do/OtherConstruct1/i HostAssoc INTEGER(4)
 i = 99
!$omp do  collapse(2)
 !DEF: /test_do/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  !DEF: /test_do/OtherConstruct1/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do j=6,10
   !DEF: /test_do/OtherConstruct1/a HostAssoc REAL(4)
   a(1,1,1) = 0.
   !DEF: /test_do/OtherConstruct1/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
   do k=11,15
    !REF: /test_do/OtherConstruct1/a
    !REF: /test_do/OtherConstruct1/k
    !REF: /test_do/OtherConstruct1/OtherConstruct1/j
    !REF: /test_do/OtherConstruct1/OtherConstruct1/i
    a(k,j,i) = 1.
   end do
  end do
 end do
!$omp end parallel
end subroutine test_do

! Rule a)
!DEF: /test_pardo (Subroutine) Subprogram
subroutine test_pardo
 implicit none
 !DEF: /test_pardo/a ObjectEntity REAL(4)
 real a(20,20,20)
 !DEF: /test_pardo/i ObjectEntity INTEGER(4)
 !DEF: /test_pardo/j ObjectEntity INTEGER(4)
 !DEF: /test_pardo/k ObjectEntity INTEGER(4)
 integer i, j, k
!$omp parallel do  collapse(2) private(k) ordered(2)
 !DEF: /test_pardo/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
   !DEF: /test_pardo/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=6,10
   !DEF: /test_pardo/OtherConstruct1/a HostAssoc REAL(4)
   a(1,1,1) = 0.
   !DEF: /test_pardo/OtherConstruct1/k (OmpPrivate) HostAssoc INTEGER(4)
   do k=11,15
    !REF: /test_pardo/OtherConstruct1/a
    !REF: /test_pardo/OtherConstruct1/k
    !REF: /test_pardo/OtherConstruct1/j
    !REF: /test_pardo/OtherConstruct1/i
    a(k,j,i) = 1.
   end do
  end do
 end do
end subroutine test_pardo

! Rule a)
!DEF: /test_taskloop (Subroutine) Subprogram
subroutine test_taskloop
 implicit none
 !DEF: /test_taskloop/a ObjectEntity REAL(4)
 real a(5,5)
 !DEF: /test_taskloop/i ObjectEntity INTEGER(4)
 !DEF: /test_taskloop/j ObjectEntity INTEGER(4)
 integer i, j
!$omp taskloop  private(j)
 !DEF: /test_taskloop/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  !DEF: /test_taskloop/OtherConstruct1/j (OmpPrivate) HostAssoc INTEGER(4)
  !REF: /test_taskloop/OtherConstruct1/i
  do j=1,i
   !DEF: /test_taskloop/OtherConstruct1/a (OmpFirstPrivate, OmpImplicit) HostAssoc REAL(4)
   !REF: /test_taskloop/OtherConstruct1/j
   !REF: /test_taskloop/OtherConstruct1/i
   a(j,i) = 3.14
  end do
 end do
!$omp end taskloop
end subroutine test_taskloop

! Rule a); OpenMP 4.5 Examples teams.2.f90
! TODO: reduction; data-mapping attributes
!DEF: /dotprod (Subroutine) Subprogram
!DEF: /dotprod/b (OmpMapTo) ObjectEntity REAL(4)
!DEF: /dotprod/c (OmpMapTo) ObjectEntity REAL(4)
!DEF: /dotprod/n ObjectEntity INTEGER(4)
!DEF: /dotprod/block_size ObjectEntity INTEGER(4)
!DEF: /dotprod/num_teams ObjectEntity INTEGER(4)
!DEF: /dotprod/block_threads ObjectEntity INTEGER(4)
subroutine dotprod (b, c, n, block_size, num_teams, block_threads)
 implicit none
 !REF: /dotprod/n
 integer n
 !REF: /dotprod/b
 !REF: /dotprod/n
 !REF: /dotprod/c
 !DEF: /dotprod/sum (OmpMapToFrom) ObjectEntity REAL(4)
 real b(n), c(n), sum
 !REF: /dotprod/block_size
 !REF: /dotprod/num_teams
 !REF: /dotprod/block_threads
 !DEF: /dotprod/i ObjectEntity INTEGER(4)
 !DEF: /dotprod/i0 ObjectEntity INTEGER(4)
 integer block_size, num_teams, block_threads, i, i0
 !REF: /dotprod/sum
 sum = 0.0e0
!$omp target  map(to:b,c)  map(tofrom:sum)
!$omp teams  num_teams(num_teams) thread_limit(block_threads) reduction(+: sum&
!$OMP&)
!$omp distribute
 !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/i0 (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 !REF: /dotprod/n
 !REF: /dotprod/block_size
 do i0=1,n,block_size
!$omp parallel do  reduction(+: sum)
  !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/i0 HostAssoc INTEGER(4)
  !DEF: /dotprod/min ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
  !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/block_size HostAssoc INTEGER(4)
  !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/n HostAssoc INTEGER(4)
  do i=i0,min(i0+block_size, n)
   !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/sum (OmpReduction) HostAssoc REAL(4)
   !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/b HostAssoc REAL(4)
   !REF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/i
   !DEF: /dotprod/OtherConstruct1/OtherConstruct1/OtherConstruct1/OtherConstruct1/c HostAssoc REAL(4)
   sum = sum+b(i)*c(i)
  end do
 end do
!$omp end teams
!$omp end target
 !REF: /dotprod/sum
 print *, sum
end subroutine dotprod

! Rule b)
! TODO: nested constructs (j, k should be private too)
!DEF: /test_simd (Subroutine) Subprogram
subroutine test_simd
 implicit none
 !DEF: /test_simd/a ObjectEntity REAL(4)
 real a(20,20,20)
 !DEF: /test_simd/i ObjectEntity INTEGER(4)
 !DEF: /test_simd/j ObjectEntity INTEGER(4)
 !DEF: /test_simd/k ObjectEntity INTEGER(4)
 integer i, j, k
!$omp parallel do simd
 !DEF: /test_simd/OtherConstruct1/i (OmpLinear, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  !DEF: /test_simd/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do j=6,10
   !DEF: /test_simd/OtherConstruct1/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
   do k=11,15
    !DEF: /test_simd/OtherConstruct1/a HostAssoc REAL(4)
    !REF: /test_simd/OtherConstruct1/k
    !REF: /test_simd/OtherConstruct1/j
    !REF: /test_simd/OtherConstruct1/i
    a(k,j,i) = 3.14
   end do
  end do
 end do
end subroutine test_simd

! Rule c)
!DEF: /test_simd_multi (Subroutine) Subprogram
subroutine test_simd_multi
 implicit none
 !DEF: /test_simd_multi/a ObjectEntity REAL(4)
 real a(20,20,20)
 !DEF: /test_simd_multi/i ObjectEntity INTEGER(4)
 !DEF: /test_simd_multi/j ObjectEntity INTEGER(4)
 !DEF: /test_simd_multi/k ObjectEntity INTEGER(4)
 integer i, j, k
!$omp parallel do simd  collapse(3)
 !DEF: /test_simd_multi/OtherConstruct1/i (OmpLastPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  !DEF: /test_simd_multi/OtherConstruct1/j (OmpLastPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do j=6,10
   !DEF: /test_simd_multi/OtherConstruct1/k (OmpLastPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
   do k=11,15
    !DEF: /test_simd_multi/OtherConstruct1/a HostAssoc REAL(4)
    !REF: /test_simd_multi/OtherConstruct1/k
    !REF: /test_simd_multi/OtherConstruct1/j
    !REF: /test_simd_multi/OtherConstruct1/i
    a(k,j,i) = 3.14
   end do
  end do
 end do
end subroutine test_simd_multi

! Rule d)
!DEF: /test_seq_loop (Subroutine) Subprogram
subroutine test_seq_loop
  implicit none
  !DEF: /test_seq_loop/i ObjectEntity INTEGER(4)
  !DEF: /test_seq_loop/j ObjectEntity INTEGER(4)
  integer i, j
  !REF: /test_seq_loop/i
  i = -1
  !REF: /test_seq_loop/j
  j = -1
  !$omp parallel
  !DEF: /test_seq_loop/OtherConstruct1/i HostAssoc INTEGER(4)
  !DEF: /test_seq_loop/OtherConstruct1/j HostAssoc INTEGER(4)
  print *, i, j
  !$omp parallel
  !DEF: /test_seq_loop/OtherConstruct1/OtherConstruct1/i HostAssoc INTEGER(4)
  !DEF: /test_seq_loop/OtherConstruct1/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  print *, i, j
  !$omp do
  !DEF: /test_seq_loop/OtherConstruct1/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
   !REF: /test_seq_loop/OtherConstruct1/OtherConstruct1/j
   do j=1,10
   end do
  end do
  !REF: /test_seq_loop/OtherConstruct1/OtherConstruct1/i
  !REF: /test_seq_loop/OtherConstruct1/OtherConstruct1/j
  print *, i, j
  !$omp end parallel
  !REF: /test_seq_loop/OtherConstruct1/i
  !REF: /test_seq_loop/OtherConstruct1/j
  print *, i, j
  !$omp end parallel
  !REF: /test_seq_loop/i
  !REF: /test_seq_loop/j
  print *, i, j
end subroutine test_seq_loop
