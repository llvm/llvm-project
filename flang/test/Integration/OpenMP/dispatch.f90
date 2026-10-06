!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

!CHECK-LABEL: define void @_QMfuncsPfoo_variant()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QMfuncsPfoo_dispatch()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QQmain()
!CHECK: call void @_QMfuncsPfoo_dispatch()
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: call void @_QMfuncsPfoo_variant()
!CHECK: br label %omp.region.cont
!CHECK: omp.region.cont:

! novariants: runtime branch between direct calls to the base and the variant.
!CHECK-LABEL: define void @test_novariants_(
!CHECK-SAME: ptr noalias %[[ARG:[0-9]+]])
!CHECK: %[[LOAD:.*]] = load i32, ptr %[[ARG]], align 4
!CHECK: %[[COND:.*]] = icmp ne i32 %[[LOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: [[NV_VARIANT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPfoo_variant()
!CHECK: [[NV_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPfoo_dispatch()
!CHECK: omp.dispatch.region:
!CHECK-NEXT: br i1 %[[COND]], label %[[NV_BASE]], label %[[NV_VARIANT]]
!CHECK: omp.region.cont:

! nocontext: runtime branch between direct calls to the base and the variant.
!CHECK-LABEL: define void @test_nocontext_(
!CHECK-SAME: ptr noalias %[[NARG:[0-9]+]])
!CHECK: %[[NLOAD:.*]] = load i32, ptr %[[NARG]], align 4
!CHECK: %[[NCOND:.*]] = icmp ne i32 %[[NLOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: [[NC_VARIANT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPfoo_variant()
!CHECK: [[NC_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPfoo_dispatch()
!CHECK: omp.dispatch.region:
!CHECK-NEXT: br i1 %[[NCOND]], label %[[NC_BASE]], label %[[NC_VARIANT]]
!CHECK: omp.region.cont:

!CHECK-LABEL: define void @test_novariants_nocontext_(
!CHECK-SAME: ptr noalias %[[C1_ARG:[0-9]+]], ptr noalias %[[C2_ARG:[0-9]+]])
!CHECK: %[[C2_LOAD:.*]] = load i32, ptr %[[C2_ARG]], align 4
!CHECK: %[[C2_COND:.*]] = icmp ne i32 %[[C2_LOAD]], 0
!CHECK: %[[C1_LOAD:.*]] = load i32, ptr %[[C1_ARG]], align 4
!CHECK: %[[C1_COND:.*]] = icmp ne i32 %[[C1_LOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: [[BOTH_DISPATCH:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPdispatch_variant()
!CHECK: [[BOTH_HOST:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPhost_variant()
!CHECK: [[BOTH_CONTEXT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: br i1 %[[C2_COND]], label %[[BOTH_HOST]], label %[[BOTH_DISPATCH]]
!CHECK: [[BOTH_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QMfuncsPbase_routine()
!CHECK: omp.dispatch.region:
!CHECK-NEXT: br i1 %[[C1_COND]], label %[[BOTH_BASE]], label %[[BOTH_CONTEXT]]
!CHECK: omp.region.cont:

! Target rewriting changes the signature of a COMPLEX result and of a
! CHARACTER(*) dummy; each branch is a direct call rewritten on its own.
!CHECK-LABEL: define void @test_complex_result_(
!CHECK-SAME: ptr noalias %[[CPLX_ARG:[0-9]+]], ptr noalias %{{[0-9]+}})
!CHECK: %[[CPLX_LOAD:.*]] = load i32, ptr %[[CPLX_ARG]], align 4
!CHECK: %[[CPLX_COND:.*]] = icmp ne i32 %[[CPLX_LOAD]], 0
!CHECK: [[CPLX_VARIANT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call {{.*}}@_QMfuncsPcomplex_variant()
!CHECK: [[CPLX_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call {{.*}}@_QMfuncsPcomplex_base()
!CHECK: omp.dispatch.region:
!CHECK-NEXT: br i1 %[[CPLX_COND]], label %[[CPLX_BASE]], label %[[CPLX_VARIANT]]

!CHECK-LABEL: define void @test_char_dummy_(
!CHECK-SAME: ptr noalias %[[CHAR_ARG:[0-9]+]])
!CHECK: %[[CHAR_LOAD:.*]] = load i32, ptr %[[CHAR_ARG]], align 4
!CHECK: %[[CHAR_COND:.*]] = icmp ne i32 %[[CHAR_LOAD]], 0
!CHECK: [[CHAR_VARIANT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK: call void @_QMfuncsPchar_variant(ptr %{{.*}}, i64 %{{.*}})
!CHECK: [[CHAR_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK: call void @_QMfuncsPchar_base(ptr %{{.*}}, i64 %{{.*}})
!CHECK: omp.dispatch.region:
!CHECK: br i1 %[[CHAR_COND]], label %[[CHAR_BASE]], label %[[CHAR_VARIANT]]

! Internal procedures take the host link as a `nest` argument, which only a
! direct call passes correctly.
!CHECK-LABEL: define void @test_internal_(
!CHECK-SAME: ptr noalias %[[INT_ARG:[0-9]+]])
!CHECK: %[[TUPLE:.*]] = alloca { ptr }
!CHECK: %[[INT_LOAD:.*]] = load i32, ptr %[[INT_ARG]], align 4
!CHECK: %[[INT_COND:.*]] = icmp ne i32 %[[INT_LOAD]], 0
!CHECK: [[INT_VARIANT:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QFtest_internalPinternal_variant(ptr %[[TUPLE]])
!CHECK: [[INT_BASE:omp\.dispatch\.region[0-9]+]]:{{[^,]*$}}
!CHECK-NEXT: call void @_QFtest_internalPinternal_base(ptr %[[TUPLE]])
!CHECK: omp.dispatch.region:
!CHECK-NEXT: br i1 %[[INT_COND]], label %[[INT_BASE]], label %[[INT_VARIANT]]
!CHECK: define internal void @_QFtest_internalPinternal_variant(ptr nest
!CHECK: define internal void @_QFtest_internalPinternal_base(ptr nest

module funcs
  implicit none

contains

  subroutine foo_variant()
    print *, "in foo_variant"
  end subroutine

  subroutine foo_dispatch()
    !$omp declare variant(foo_dispatch:foo_variant) match(construct={dispatch})
    print *, "in foo_dispatch"
  end subroutine

  subroutine dispatch_variant()
    print *, "in dispatch_variant"
  end subroutine

  subroutine host_variant()
    print *, "in host_variant"
  end subroutine

  subroutine base_routine()
    !$omp declare variant(base_routine:dispatch_variant) match(construct={dispatch})
    !$omp declare variant(base_routine:host_variant) match(device={kind(host)})
    print *, "in base_routine"
  end subroutine

  complex function complex_variant() result(r)
    r = (2.0, 2.0)
  end function

  complex function complex_base() result(r)
    !$omp declare variant(complex_base:complex_variant) match(construct={dispatch})
    r = (1.0, 1.0)
  end function

  subroutine char_variant(c)
    character(*), intent(in) :: c
    print *, "in char_variant ", c
  end subroutine

  subroutine char_base(c)
    !$omp declare variant(char_base:char_variant) match(construct={dispatch})
    character(*), intent(in) :: c
    print *, "in char_base ", c
  end subroutine

end module funcs

program dispatch_test
  use funcs
  implicit none

  call foo_dispatch()

  !$omp dispatch
  call foo_dispatch()

end program

subroutine test_novariants(cond)
  use funcs
  implicit none
  logical :: cond

  !$omp dispatch novariants(cond)
  call foo_dispatch()

end subroutine

subroutine test_nocontext(cond)
  use funcs
  implicit none
  logical :: cond

  !$omp dispatch nocontext(cond)
  call foo_dispatch()

end subroutine

subroutine test_novariants_nocontext(c1, c2)
  use funcs
  implicit none
  logical :: c1, c2

  !$omp dispatch novariants(c1) nocontext(c2)
  call base_routine()

end subroutine

subroutine test_complex_result(cond, z)
  use funcs
  implicit none
  logical :: cond
  complex :: z

  !$omp dispatch novariants(cond)
  z = complex_base()

end subroutine

subroutine test_char_dummy(cond)
  use funcs
  implicit none
  logical :: cond

  !$omp dispatch nocontext(cond)
  call char_base("abc")

end subroutine

subroutine test_internal(cond)
  implicit none
  logical :: cond
  integer :: captured

  captured = 1
  !$omp dispatch novariants(cond)
  call internal_base()

contains
  subroutine internal_variant()
    print *, captured + 1
  end subroutine

  subroutine internal_base()
    !$omp declare variant(internal_base:internal_variant) match(construct={dispatch})
    print *, captured
  end subroutine
end subroutine
