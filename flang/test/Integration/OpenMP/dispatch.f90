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

! novariants: runtime select of base/variant address, then indirect call.
!CHECK-LABEL: define void @test_novariants_(
!CHECK-SAME: ptr noalias %[[ARG:[0-9]+]])
!CHECK: %[[LOAD:.*]] = load i32, ptr %[[ARG]], align 4
!CHECK: %[[COND:.*]] = icmp ne i32 %[[LOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: %[[TARGET:.*]] = select i1 %[[COND]], ptr @_QMfuncsPfoo_dispatch, ptr @_QMfuncsPfoo_variant
!CHECK: call void %[[TARGET]]()
!CHECK: br label %omp.region.cont
!CHECK: omp.region.cont:

! nocontext: runtime select of base/variant address, then indirect call.
!CHECK-LABEL: define void @test_nocontext_(
!CHECK-SAME: ptr noalias %[[NARG:[0-9]+]])
!CHECK: %[[NLOAD:.*]] = load i32, ptr %[[NARG]], align 4
!CHECK: %[[NCOND:.*]] = icmp ne i32 %[[NLOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: %[[NTARGET:.*]] = select i1 %[[NCOND]], ptr @_QMfuncsPfoo_dispatch, ptr @_QMfuncsPfoo_variant
!CHECK: call void %[[NTARGET]]()
!CHECK: br label %omp.region.cont
!CHECK: omp.region.cont:

!CHECK-LABEL: define void @test_novariants_nocontext_(
!CHECK-SAME: ptr noalias %[[C1_ARG:[0-9]+]], ptr noalias %[[C2_ARG:[0-9]+]])
!CHECK: %[[C2_LOAD:.*]] = load i32, ptr %[[C2_ARG]], align 4
!CHECK: %[[C2_COND:.*]] = icmp ne i32 %[[C2_LOAD]], 0
!CHECK: %[[C1_LOAD:.*]] = load i32, ptr %[[C1_ARG]], align 4
!CHECK: %[[C1_COND:.*]] = icmp ne i32 %[[C1_LOAD]], 0
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: %[[CONTEXT_TARGET:.*]] = select i1 %[[C2_COND]], ptr @_QMfuncsPhost_variant, ptr @_QMfuncsPdispatch_variant
!CHECK: %[[BOTH_TARGET:.*]] = select i1 %[[C1_COND]], ptr @_QMfuncsPbase_routine, ptr %[[CONTEXT_TARGET]]
!CHECK-NEXT: call void %[[BOTH_TARGET]]()
!CHECK-NEXT: br label %omp.region.cont
!CHECK: omp.region.cont:

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

!CHECK-LABEL: define void @test_external_novariants_(
!CHECK-SAME: ptr noalias %[[EXT_COND_ARG:[0-9]+]])
!CHECK: %[[EXT_LOAD:.*]] = load i32, ptr %[[EXT_COND_ARG]], align 4
!CHECK: %[[EXT_COND:.*]] = icmp ne i32 %[[EXT_LOAD]], 0
!CHECK: %[[EXT_TARGET:.*]] = select i1 %[[EXT_COND]], ptr @external_base_, ptr @external_variant_
!CHECK-NEXT: call void %[[EXT_TARGET]]()
subroutine test_external_novariants(cond)
  implicit none
  logical :: cond
  interface
    subroutine external_variant()
    end subroutine
    subroutine external_base()
      import :: external_variant
      !$omp declare variant(external_base:external_variant) match(construct={dispatch})
    end subroutine
  end interface

  !$omp dispatch novariants(cond)
  call external_base()
end subroutine

!CHECK-LABEL: define i32 @test_external_nocontext_(
!CHECK-SAME: ptr noalias %[[EXT_NCOND_ARG:[0-9]+]], ptr noalias %[[EXT_VALUE_ARG:[0-9]+]])
!CHECK: %[[EXT_NLOAD:.*]] = load i32, ptr %[[EXT_NCOND_ARG]], align 4
!CHECK: %[[EXT_NCOND:.*]] = icmp ne i32 %[[EXT_NLOAD]], 0
!CHECK: %[[EXT_VALUE:.*]] = load i32, ptr %[[EXT_VALUE_ARG]], align 4
!CHECK: %[[EXT_NTARGET:.*]] = select i1 %[[EXT_NCOND]], ptr @external_host_func_, ptr @external_dispatch_func_
!CHECK-NEXT: %[[EXT_RESULT:.*]] = call i32 %[[EXT_NTARGET]](i32 %[[EXT_VALUE]])
!CHECK: store i32 %[[EXT_RESULT]], ptr %[[EXT_RESULT_ADDR:.*]], align 4
!CHECK: %[[EXT_RETURN:.*]] = load i32, ptr %[[EXT_RESULT_ADDR]], align 4
!CHECK: ret i32 %[[EXT_RETURN]]
integer function test_external_nocontext(cond, value) result(res)
  implicit none
  logical :: cond
  integer :: value
  interface
    integer function external_dispatch_func(value)
      integer, value :: value
    end function
    integer function external_host_func(value)
      integer, value :: value
    end function
    integer function external_base_func(value) result(output)
      import :: external_dispatch_func, external_host_func
      integer, value :: value
      !$omp declare variant(external_base_func:external_dispatch_func) match(construct={dispatch})
      !$omp declare variant(external_base_func:external_host_func) match(device={kind(host)})
    end function
  end interface

  !$omp dispatch nocontext(cond)
  res = external_base_func(value)
end function

!CHECK-DAG: declare void @external_variant_()
!CHECK-DAG: declare void @external_base_()
!CHECK-DAG: declare i32 @external_dispatch_func_(i32)
!CHECK-DAG: declare i32 @external_host_func_(i32)
