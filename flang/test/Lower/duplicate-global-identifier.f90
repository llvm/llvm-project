! Semantics does not diagnose a BIND(C) variable and a BIND(C) procedure that
! share one global identifier when they are declared in different scopes.
! Lowering then creates two symbols with the same name; the FIR verifier
! must reject the module, and the bridge must not assert while declaring the
! second one.
! RUN: not %flang_fc1 -emit-fir %s 2>&1 | FileCheck %s

module clash_mod
  use, intrinsic :: iso_c_binding, only : c_int
  integer(c_int), bind(c, name='clash') :: counter
end module

subroutine caller
  use clash_mod
  interface
    subroutine clash() bind(c)
    end subroutine
  end interface
  counter = 1
  call clash()
end subroutine

! CHECK: error: {{.*}}redefinition of symbol named 'clash'
! CHECK: error: verification of lowering to FIR failed
