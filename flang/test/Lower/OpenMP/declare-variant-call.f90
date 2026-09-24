! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! Lowering tests for DECLARE VARIANT callee resolution at call sites.
! The declarative directive is not lowered; variant selection rewrites
! procedure calls inside matching OpenMP regions. The base and its variant are
! sibling internal procedures of the calling host, so the variant is accessible
! at every reference to the base.

subroutine test_sequential_vs_parallel
  call base()
  !$omp parallel
  call base()
  !$omp end parallel
contains
  subroutine base
    !$omp declare variant (base:vsub) match (construct={parallel})
  end subroutine base
  subroutine vsub
  end subroutine vsub
end subroutine test_sequential_vs_parallel

! CHECK-LABEL: func.func @_QPtest_sequential_vs_parallel
! CHECK: fir.call @_QFtest_sequential_vs_parallelPbase(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_sequential_vs_parallelPvsub(){{.*}}: () -> ()

subroutine test_teams_vs_parallel
  !$omp parallel
  call base2()
  !$omp end parallel
  !$omp teams
  call base2()
  !$omp end teams
contains
  subroutine base2
    !$omp declare variant (base2:vsub_par) match (construct={parallel})
    !$omp declare variant (base2:vsub_teams) match (construct={teams})
  end subroutine base2
  subroutine vsub_par
  end subroutine vsub_par
  subroutine vsub_teams
  end subroutine vsub_teams
end subroutine test_teams_vs_parallel

! CHECK-LABEL: func.func @_QPtest_teams_vs_parallel
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_teams_vs_parallelPvsub_par(){{.*}}: () -> ()
! CHECK: omp.teams
! CHECK: fir.call @_QFtest_teams_vs_parallelPvsub_teams(){{.*}}: () -> ()

subroutine test_user_condition_false
  !$omp parallel
  call base3()
  !$omp end parallel
contains
  subroutine base3
    !$omp declare variant (base3:vsub) match (user={condition(.false.)})
  end subroutine base3
  subroutine vsub
  end subroutine vsub
end subroutine test_user_condition_false

! CHECK-LABEL: func.func @_QPtest_user_condition_false
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_user_condition_falsePbase3(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFtest_user_condition_falsePvsub

subroutine test_user_condition_true
  !$omp parallel
  call base4()
  !$omp end parallel
contains
  subroutine base4
    !$omp declare variant (base4:vsub) match (user={condition(.true.)})
  end subroutine base4
  subroutine vsub
  end subroutine vsub
end subroutine test_user_condition_true

! CHECK-LABEL: func.func @_QPtest_user_condition_true
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_user_condition_truePvsub(){{.*}}: () -> ()

! With the base name omitted, the enclosing procedure (the internal subprogram
! omit_base) is the base; its sibling internal vsub is the variant.

subroutine test_omitted_base_name
  call omit_base()
  !$omp parallel
  call omit_base()
  !$omp end parallel
contains
  subroutine omit_base
    !$omp declare variant (vsub) match (construct={parallel})
  end subroutine omit_base
  subroutine vsub
  end subroutine vsub
end subroutine test_omitted_base_name

! CHECK-LABEL: func.func @_QPtest_omitted_base_name
! CHECK: fir.call @_QFtest_omitted_base_namePomit_base(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_omitted_base_namePvsub(){{.*}}: () -> ()

subroutine test_call_with_args
  integer :: x
  x = 1
  call base5(x)
  !$omp parallel
  call base5(x)
  !$omp end parallel
contains
  subroutine base5(n)
    integer, intent(in) :: n
    !$omp declare variant (base5:vsub) match (construct={parallel})
  end subroutine base5
  subroutine vsub(n)
    integer, intent(in) :: n
  end subroutine vsub
end subroutine test_call_with_args

! CHECK-LABEL: func.func @_QPtest_call_with_args
! CHECK: fir.call @_QFtest_call_with_argsPbase5(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_call_with_argsPvsub(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()

subroutine test_no_variant_recorded
  call plain()
  !$omp parallel
  call plain()
  !$omp end parallel
contains
  subroutine plain
  end subroutine plain
end subroutine test_no_variant_recorded

! CHECK-LABEL: func.func @_QPtest_no_variant_recorded
! CHECK: fir.call @_QFtest_no_variant_recordedPplain(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_no_variant_recordedPplain(){{.*}}: () -> ()

subroutine test_nested_parallel
  !$omp parallel
    !$omp parallel
    call base6()
    !$omp end parallel
  !$omp end parallel
contains
  subroutine base6
    !$omp declare variant (base6:vsub) match (construct={parallel})
  end subroutine base6
  subroutine vsub
  end subroutine vsub
end subroutine test_nested_parallel

! CHECK-LABEL: func.func @_QPtest_nested_parallel
! CHECK: omp.parallel
! CHECK: omp.parallel
! CHECK: fir.call @_QFtest_nested_parallelPvsub(){{.*}}: () -> ()

subroutine test_parallel_do
  integer :: i
  !$omp parallel do
  do i = 1, 2
    call base7(i)
  end do
contains
  subroutine base7(n)
    integer, intent(in) :: n
    !$omp declare variant (base7:vsub) match (construct={parallel})
  end subroutine base7
  subroutine vsub(n)
    integer, intent(in) :: n
  end subroutine vsub
end subroutine test_parallel_do

! CHECK-LABEL: func.func @_QPtest_parallel_do
! CHECK: fir.call @_QFtest_parallel_doPvsub(%{{.*}}){{.*}}: (!fir.ref<i32>) -> ()

subroutine test_target_construct
  call base8()
  !$omp target
  call base8()
  !$omp end target
contains
  subroutine base8
    !$omp declare variant (base8:vsub) match (construct={target})
  end subroutine base8
  subroutine vsub
  end subroutine vsub
end subroutine test_target_construct

! CHECK-LABEL: func.func @_QPtest_target_construct
! CHECK: fir.call @_QFtest_target_constructPbase8(){{.*}}: () -> ()
! CHECK: omp.target
! CHECK: fir.call @_QFtest_target_constructPvsub(){{.*}}: () -> ()

! DECLARE VARIANT substitutes the variant only at an actual call to the base.
! Taking the base's address to pass it as an actual argument is not a call, so
! it must still reference the base, even inside a matching context.

subroutine test_reference_vs_call
  !$omp parallel
  call apply(base9)
  call base9()
  !$omp end parallel
contains
  subroutine base9
    !$omp declare variant (base9:vsub) match (construct={parallel})
  end subroutine base9
  subroutine vsub
  end subroutine vsub
  subroutine apply(f)
    external :: f
    call f()
  end subroutine apply
end subroutine test_reference_vs_call

! CHECK-LABEL: func.func @_QPtest_reference_vs_call
! CHECK: omp.parallel
! CHECK: fir.address_of(@_QFtest_reference_vs_callPbase9)
! CHECK: fir.call @_QFtest_reference_vs_callPapply
! CHECK: fir.call @_QFtest_reference_vs_callPvsub(){{.*}}: () -> ()

! DECLARE VARIANT applies only to a direct call to the base in our
! implementation.

subroutine test_dummy_not_replaced
  call indirect_caller(base10)
contains
  subroutine base10
    !$omp declare variant (base10:vsub) match (construct={parallel})
  end subroutine base10
  subroutine vsub
  end subroutine vsub
  subroutine indirect_caller(f)
    procedure() :: f
    !$omp parallel
    call f()
    !$omp end parallel
  end subroutine indirect_caller
end subroutine test_dummy_not_replaced

! CHECK-LABEL: func.func {{.*}}@_QFtest_dummy_not_replacedPindirect_caller
! CHECK: omp.parallel
! CHECK: fir.call %{{[0-9]+}}(){{.*}}: () -> ()
