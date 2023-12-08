! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp %s -o - | FileCheck %s

subroutine sb(n)
implicit none

integer :: n
integer :: arr(n,n), brr(n,n), crr(n,n)
integer :: arr_single(n),arr_quad(n,n,n,n)
integer :: i,j,k,l,tmp,tmp1,tmp2

! CHECK:---

!Simple check with nowait
!$omp do
do i = 1, n
    arr_single(i) = i
end do
!$omp end do nowait
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-6]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''

!Check for no effects on loop without nowait
!$omp do
do i = 1, n
    arr_single(i) = i
end do
!$omp end do
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-6]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:         []

!Check with another construct nested inside loop with nowait
!$omp parallel shared(arr)
!$omp do
do i = 1, n
!$omp critical
    arr_single(i) = i
!$omp end critical
end do
!$omp end do nowait
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-7]]
! CHECK-NEXT:  construct:       critical
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-13]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-20]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     arr

!Check with back to back loops (one with nowait) inside a parallel construct
!$omp parallel shared(arr)
!$omp do
do i=1,10
    arr(i,j) = i+j
end do
!$omp end do nowait
!$omp do schedule(guided)
do j=1,10
end do
!$omp end do
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-11]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-12]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      schedule
! CHECK-NEXT:      details:     guided
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-24]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     arr


!Check nested parallel do loops with a nowait outside
!$omp parallel shared(arr)
!$omp do
do i=1,10
    arr_single(i)=i
    !$omp parallel
    !$omp do
    do j=1,10
        !$omp critical
        arr(i,j) = i+j
        !$omp end critical
    end do
    !$omp end do
    !$omp end parallel
end do
!$omp end do nowait
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-10]]
! CHECK-NEXT:  construct:       critical
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-16]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-21]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-28]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-35]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     arr

!Check nested parallel do loops with a nowait inside
!$omp parallel shared(arr)
!$omp do
do i=1,10
    arr_single(i)=i
    !$omp parallel
    !$omp do
    do j=1,10
        !$omp critical
        arr(i,j) = i+j
        !$omp end critical
    end do
    !$omp end do nowait
    !$omp end parallel
end do
!$omp end do
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-10]]
! CHECK-NEXT:  construct:       critical
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-16]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-23]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-30]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-35]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     arr

!Check nested parallel do loops with a nowait inside
!$omp parallel
!$omp do
do i=1,10
    arr_single(i)=i
    !$omp parallel shared(arr_quad)
    !$omp do schedule(dynamic)
    do j=1,10
        !$omp parallel
        !$omp do
        do k=1,10
            !$omp parallel
            !$omp do
            do l=1,10
                arr_quad(i,j,k,l) = i+j+k+l
            end do
            !$omp end do nowait
            !$omp end parallel
        end do
        !$omp end do
        !$omp end parallel
    end do
    !$omp end do nowait
    !$omp end parallel
end do
!$omp end do
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-16]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-23]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-29]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-34]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-40]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:    - clause:      schedule
! CHECK-NEXT:      details:     dynamic
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-49]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      shared
! CHECK-NEXT:      details:     arr_quad
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-58]]
! CHECK-NEXT:  construct:       do
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-63]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []


!Check a do simd with nowait
!$omp do simd private(tmp)
do j = 1,n
    do i = 1,n
        tmp = arr(i,j) + brr(i,j)
        crr(i,j) = tmp
    end do
end do
!$omp end do simd nowait
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-9]]
! CHECK-NEXT:  construct:       do simd
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:    - clause:      private
! CHECK-NEXT:      details:     tmp


!test nowait on non-do construct
!$omp parallel
!$omp single
tmp1 = i+j
!$omp end single

!$omp single
tmp2 = i-j
!$omp end single nowait
!$omp end parallel
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-9]]
! CHECK-NEXT:  construct:       single
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-9]]
! CHECK-NEXT:  construct:       single
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      nowait
! CHECK-NEXT:      details:     ''
! CHECK-NEXT:- file:            '{{[^"]*}}omp-nowait.f90'
! CHECK-NEXT:  line:            [[@LINE-20]]
! CHECK-NEXT:  construct:       parallel
! CHECK-NEXT:  clauses:         []

end subroutine

! CHECK-NEXT:...
