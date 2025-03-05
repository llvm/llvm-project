! Test fir.host_sym attribute to retain link between internal
! and host procedure in FIR even when BIND(C) is involved.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -o - %s | fir-opt -external-name-interop -o - |FileCheck %s --check-prefix=AFTER_RENAME_PASS

subroutine foo() bind(c, name="some_c_name")
  call bar()
contains
 subroutine bar()
 end subroutine
end subroutine
! CHECK: func.func @some_c_name()
! CHECK: func.func private @_QFfooPbar() attributes {fir.host_symbol = @some_c_name, llvm.linkage = #llvm.linkage<internal>}
! AFTER_RENAME_PASS: func.func @some_c_name()
! AFTER_RENAME_PASS: func.func private @_QFfooPbar() attributes {fir.host_symbol = @some_c_name, llvm.linkage = #llvm.linkage<internal>}

subroutine notbindc()
  call bar()
contains
 subroutine bar()
 end subroutine
end subroutine
! CHECK: func.func @_QPnotbindc()
! CHECK: func.func private @_QFnotbindcPbar() attributes {fir.host_symbol = @_QPnotbindc, llvm.linkage = #llvm.linkage<internal>}
! AFTER_RENAME_PASS: func.func @notbindc_() attributes {fir.internal_name = "_QPnotbindc"}
! AFTER_RENAME_PASS: func.func private @_QFnotbindcPbar() attributes {fir.host_symbol = @notbindc_, llvm.linkage = #llvm.linkage<internal>}


! Main program
call bar()
contains
 subroutine bar()
 end subroutine
end
! CHECK: func.func @_QQmain()
! CHECK: func.func private @_QFPbar() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>}
! AFTER_RENAME_PASS: func.func @_QQmain()
! AFTER_RENAME_PASS: func.func private @_QFPbar() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>}
