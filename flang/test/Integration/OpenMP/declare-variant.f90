!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Test that declare variant with construct context selectors correctly
! resolves variant calls in the generated LLVM IR.

module m
contains
  subroutine variant_parallel()
  end subroutine
  subroutine variant_teams()
  end subroutine
  subroutine base_parallel()
    !$omp declare variant(variant_parallel) match(construct={parallel})
  end subroutine
  subroutine base_teams()
    !$omp declare variant(variant_teams) match(construct={teams})
  end subroutine
end module m

! CHECK-LABEL: define internal void @test_parallel_..omp_par
! CHECK:         call void {{.*}}variant_parallel()
subroutine test_parallel()
  use m
  !$omp parallel
    call base_parallel()
  !$omp end parallel
end subroutine

! CHECK-LABEL: define internal void @test_teams_..omp_par
! CHECK:         call void {{.*}}variant_teams()
subroutine test_teams()
  use m
  !$omp teams
    call base_teams()
  !$omp end teams
end subroutine

! Verify that outside a matching construct, the base call is preserved.
! CHECK-LABEL: define void {{.*}}test_no_construct{{.*}}
! CHECK:         call void {{.*}}base_parallel()
! CHECK-NOT:     call void {{.*}}variant_parallel
subroutine test_no_construct()
  use m
  call base_parallel()
end subroutine

! Test declare variant with user={condition(expr)} only.
module m_condition
  implicit none
  logical :: use_variant = .true.
contains
  subroutine foo_variant_cond()
  end subroutine
  subroutine foo_cond()
    !$omp declare variant(foo_variant_cond) match(user={condition(use_variant)})
  end subroutine
end module m_condition

! CHECK-LABEL: define void {{.*}}foo_cond()
! CHECK:         br i1 %{{.*}}, label %[[THEN:.*]], label %[[ELSE:.*]]
! CHECK:       [[THEN]]:
! CHECK:         call void {{.*}}foo_variant_cond()
! CHECK:         ret void
! CHECK:       [[ELSE]]:
! CHECK:         ret void
subroutine test_condition()
  use m_condition
  call foo_cond()
end subroutine

! Test declare variant with both construct={dispatch} and user={condition(expr)}.
module m_both
  implicit none
  logical :: use_variant_both = .true.
contains
  subroutine bar_variant()
  end subroutine
  subroutine bar()
    !$omp declare variant(bar_variant) &
    !$omp& match(construct={dispatch}, user={condition(use_variant_both)})
  end subroutine
end module m_both

! CHECK-LABEL: define void {{.*}}test_dispatch_both{{.*}}
! CHECK:         br i1 %{{.*}}, label %[[THEN2:.*]], label %[[ELSE2:.*]]
! CHECK:       [[THEN2]]:
! CHECK:         call void {{.*}}bar_variant()
! CHECK:       [[ELSE2]]:
! CHECK:         call void {{.*}}bar()
subroutine test_dispatch_both()
  use m_both
  !$omp dispatch
  call bar()
  !$omp end dispatch
end subroutine

! Verify that without dispatch, the base call is preserved.
! CHECK-LABEL: define void {{.*}}test_no_dispatch_both{{.*}}
! CHECK:         call void {{.*}}bar()
! CHECK-NOT:     call void {{.*}}bar_variant
subroutine test_no_dispatch_both()
  use m_both
  call bar()
end subroutine
