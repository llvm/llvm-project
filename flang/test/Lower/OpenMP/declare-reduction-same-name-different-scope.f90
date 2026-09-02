! Test that two user-defined reductions sharing the same name but declared in
! different scopes lower to distinct omp.declare_reduction operations, and that
! a reduction clause refers to the declaration visible in its own scope rather
! than one leaking in from another scope (issue #181270).

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
contains
  subroutine dummy
!$omp declare reduction (a:integer:omp_out=omp_out+omp_in) initializer(omp_priv=10000)
  end subroutine dummy

  subroutine test
!$omp declare reduction (a:integer:omp_out=omp_out+omp_in) initializer(omp_priv=0)
    integer::x1,i
    x1=0
!$omp parallel do reduction(a:x1)
    do i=1,10
       x1=x1+1
    end do
!$omp end parallel do
  end subroutine test
end module m

! CHECK: omp.declare_reduction @[[TEST_RED:_QQMmFtesta]] : i32 init {
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: omp.yield(%[[C0]] : i32)

! CHECK: omp.declare_reduction @[[DUMMY_RED:_QQMmFdummya]] : i32 init {
! CHECK: %[[C10000:.*]] = arith.constant 10000 : i32
! CHECK: omp.yield(%[[C10000]] : i32)

! CHECK-LABEL: func.func @_QMmPtest()
! CHECK: omp.wsloop {{.*}}reduction(@[[TEST_RED]] %{{.*}} -> %{{.*}} : !fir.ref<i32>)
