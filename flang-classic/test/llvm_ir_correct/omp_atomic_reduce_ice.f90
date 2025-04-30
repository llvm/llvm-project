
! RUN: %flang -fopenmp -Hy,69,0x1000 -S -emit-llvm %s -o - | FileCheck %s
! RUN: %flang -i8 -fopenmp -Hy,69,0x1000 -S -emit-llvm %s -o - | FileCheck %s

subroutine reduce()
integer :: j
logical(kind=8) :: error_status = .FALSE.
!$omp parallel do reduction(.or.: error_status)
do j=1,100
end do
!$omp end parallel do
end subroutine
! //CHECK: atomicrmw

subroutine atomic_logical_1(n,val)
logical(kind=1) :: lg = .FALSE.
integer :: n, val, i
!$omp parallel
do i=1,n
!$omp atomic
lg = lg .or. val==n
end do
!$omp end parallel
end subroutine
! //CHECK: atomicrmw

subroutine atomic_logical_8(n,val)
logical(kind=8) :: lg = .FALSE.
integer :: n, val, i
!$omp parallel
do i=1,n
!$omp atomic
lg = lg .or. val==n
end do
!$omp end parallel
end subroutine
! //CHECK: atomicrmw

subroutine atomic_integer_1(n,val)
integer(kind=1) :: lg, val
integer :: n, i
!$omp parallel
do i=1,n
!$omp atomic
lg = lg + val
end do
!$omp end parallel
end subroutine
! //CHECK: atomicrmw
