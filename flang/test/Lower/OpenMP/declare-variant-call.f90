! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! Lowering tests for DECLARE VARIANT callee resolution at call sites.
! The declarative directive is not lowered; variant selection rewrites
! procedure calls inside matching OpenMP regions.

subroutine test_sequential_vs_parallel
  call base()
  !$omp parallel
  call base()
  !$omp end parallel
end subroutine test_sequential_vs_parallel

subroutine base
  !$omp declare variant (base:vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine
end subroutine base

! CHECK-LABEL: func.func @_QPtest_sequential_vs_parallel
! CHECK: fir.call @_QPbase(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFbasePvsub(){{.*}}: () -> ()

subroutine test_teams_vs_parallel
  !$omp parallel
  call base2()
  !$omp end parallel
  !$omp teams
  call base2()
  !$omp end teams
end subroutine test_teams_vs_parallel

subroutine base2
  !$omp declare variant (base2:vsub_par) match (construct={parallel})
  !$omp declare variant (base2:vsub_teams) match (construct={teams})
contains
  subroutine vsub_par
  end subroutine
  subroutine vsub_teams
  end subroutine
end subroutine base2

! CHECK-LABEL: func.func @_QPtest_teams_vs_parallel
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase2Pvsub_par(){{.*}}: () -> ()
! CHECK: omp.teams
! CHECK: fir.call @_QFbase2Pvsub_teams(){{.*}}: () -> ()

subroutine test_user_condition_false
  !$omp parallel
  call base3()
  !$omp end parallel
end subroutine test_user_condition_false

subroutine base3
  !$omp declare variant (base3:vsub) match (user={condition(.false.)})
contains
  subroutine vsub
  end subroutine
end subroutine base3

! CHECK-LABEL: func.func @_QPtest_user_condition_false
! CHECK: omp.parallel
! CHECK: fir.call @_QPbase3(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase3Pvsub

subroutine test_user_condition_true
  !$omp parallel
  call base4()
  !$omp end parallel
end subroutine test_user_condition_true

subroutine base4
  !$omp declare variant (base4:vsub) match (user={condition(.true.)})
contains
  subroutine vsub
  end subroutine
end subroutine base4

! CHECK-LABEL: func.func @_QPtest_user_condition_true
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase4Pvsub(){{.*}}: () -> ()

subroutine test_omitted_base_name
  !$omp parallel
  call host()
  !$omp end parallel
end subroutine test_omitted_base_name

subroutine host
  !$omp declare variant (vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine
end subroutine host

! CHECK-LABEL: func.func @_QPtest_omitted_base_name
! CHECK: omp.parallel
! CHECK: fir.call @_QFhostPvsub(){{.*}}: () -> ()

subroutine test_call_with_args
  integer :: x
  x = 1
  call base5(x)
  !$omp parallel
  call base5(x)
  !$omp end parallel
end subroutine test_call_with_args

subroutine base5(n)
  integer, intent(in) :: n
  !$omp declare variant (base5:vsub) match (construct={parallel})
contains
  subroutine vsub(n)
    integer, intent(in) :: n
  end subroutine
end subroutine base5

! CHECK-LABEL: func.func @_QPtest_call_with_args
! CHECK: fir.call @_QPbase5(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase5Pvsub(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()

subroutine test_no_variant_recorded
  call plain()
  !$omp parallel
  call plain()
  !$omp end parallel
end subroutine test_no_variant_recorded

subroutine plain
end subroutine plain

! CHECK-LABEL: func.func @_QPtest_no_variant_recorded
! CHECK: fir.call @_QPplain(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QPplain(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFplainP

subroutine test_nested_parallel
  !$omp parallel
    !$omp parallel
    call base6()
    !$omp end parallel
  !$omp end parallel
end subroutine test_nested_parallel

subroutine base6
  !$omp declare variant (base6:vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine
end subroutine base6

! CHECK-LABEL: func.func @_QPtest_nested_parallel
! CHECK: omp.parallel
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase6Pvsub(){{.*}}: () -> ()

subroutine test_parallel_do
  integer :: i
  !$omp parallel do
  do i = 1, 2
    call base7(i)
  end do
end subroutine test_parallel_do

subroutine base7(n)
  integer, intent(in) :: n
  !$omp declare variant (base7:vsub) match (construct={parallel})
contains
  subroutine vsub(n)
    integer, intent(in) :: n
  end subroutine
end subroutine base7

! CHECK-LABEL: func.func @_QPtest_parallel_do
! CHECK: fir.call @_QFbase7Pvsub(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()

subroutine test_target_construct
  call base8()
  !$omp target
  call base8()
  !$omp end target
end subroutine test_target_construct

subroutine base8
  !$omp declare variant (base8:vsub) match (construct={target})
contains
  subroutine vsub
  end subroutine
end subroutine base8

! CHECK-LABEL: func.func @_QPtest_target_construct
! CHECK: fir.call @_QPbase8(){{.*}}: () -> ()
! CHECK: omp.target
! CHECK: fir.call @_QFbase8Pvsub(){{.*}}: () -> ()
