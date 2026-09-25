! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that ASYNCHRONOUS/VOLATILE statements in a submodule correctly
! host-associate variables from the ancestor module rather than creating
! new local symbols. GitHub issue #208362.
!
! Before the fix, n and k were given fresh submodule-scoped symbols
! (_QMm1SsubmodEn / _QMm1SsubmodEk). Verify they resolve to the module
! globals and that no submodule-mangled name appears in the function body.

module m1
  integer :: n = 1, k = 2
  interface
    module subroutine sub()
    end subroutine
  end interface
end module m1

submodule(m1) submod
  volatile :: n
  asynchronous :: k
contains
  ! CHECK-LABEL: func @_QMm1Psub
  ! No submodule-mangled entity should appear inside the subroutine.
  ! CHECK-NOT: _QMm1Ssubmod
  ! n and k must resolve to the module globals.
  ! CHECK-DAG: %[[N:.*]] = fir.address_of(@_QMm1En) : !fir.ref<i32>
  ! CHECK-DAG: %[[K:.*]] = fir.address_of(@_QMm1Ek) : !fir.ref<i32>
  ! CHECK-DAG: hlfir.declare %[[N]] {uniq_name = "_QMm1En"}
  ! CHECK-DAG: hlfir.declare %[[K]] {uniq_name = "_QMm1Ek"}
  ! FIXME: The volatile/asynchronous fortran_attrs are not propagated to the
  ! hlfir.declare or fir.ref type for host-associated variables (this affects
  ! regular subprograms too, not just submodules).
  ! See https://github.com/llvm/llvm-project/issues/208588.
  module subroutine sub()
    implicit none
    if (n /= 1) print *, 'Error n=', n
    if (k /= 2) print *, 'Error k=', k
  end subroutine
end submodule submod

! CHECK: fir.global @_QMm1Ek : i32
! CHECK: fir.global @_QMm1En : i32
