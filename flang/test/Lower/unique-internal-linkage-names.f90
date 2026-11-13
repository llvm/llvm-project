! Test that -funique-internal-linkage-names appends a .__uniq. hash suffix
! to internal procedures at the FIR level.

! RUN: %flang_fc1 -emit-fir -funique-internal-linkage-names -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPtest
! CHECK: fir.call @_QFtestPfoo.__uniq.{{[0-9]+}}

! CHECK: func.func private @_QFtestPfoo.__uniq.{{[0-9]+}}
! CHECK-SAME: attributes {fir.host_symbol = @_QPtest, llvm.linkage = #llvm.linkage<internal>}

subroutine test()
  call foo()
contains
  subroutine foo()
  end subroutine
end subroutine
