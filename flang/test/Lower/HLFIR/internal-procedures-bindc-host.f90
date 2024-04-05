! Test fir.host_sym attribute to retain link between internal
! and host procedure in FIR even when BIND(C) is involved.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine foo() bind(c, name="some_c_name")
  call bar()
contains
 subroutine bar()
 end subroutine
end subroutine
! CHECK: func.func @some_c_name()
! CHECK: func.func private @_QFfooPbar() attributes {fir.host_symbol = @some_c_name, llvm.linkage = #llvm.linkage<internal>}
