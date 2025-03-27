! RUN: %flang_fc1 -fopenmp -E %s 2>&1 | FileCheck %s
! CHECK: call foo(0.)
! CHECK: call foo(1.)
! CHECK: call foo(2.)
! CHECK: call foo(3.)
! CHECK: !$omp parallel do default(none) private(j)
! CHECK: !$omp end parallel do
call foo( &
# 100 "bar.h"
         & 0.)
call foo( &
# 101 "bar.h"
         1.)
call foo( &
# 102 "bar.h"
         & 2. &
    & )
call foo( &
# 103 "bar.h"
         & 3. &
    )
!$omp parallel do &
#ifdef undef
!$omp garbage &
#else
!$omp default(none) &
#endif
!$omp private(j)
  do j=1,100
  end do
!$omp end &
# 104 "bar.h"
!$omp parallel do
end
