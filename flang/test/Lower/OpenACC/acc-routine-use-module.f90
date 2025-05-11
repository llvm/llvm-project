! RUN: rm -fr %t && mkdir -p %t && cd %t
! RUN: bbc -fopenacc -emit-fir %S/acc-module-definition.f90
! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! This test module is based off of flang/test/Lower/use_module.f90
! The first runs ensures the module file is generated.

module use_mod1
  use mod1
  contains
    !CHECK: acc.routine @acc_routine_0 func(@_QMmod1Pcallee) seq
    !CHECK: func.func @_QMuse_mod1Pcaller
    !CHECK-SAME {
    subroutine caller(aa)
      integer :: aa
      !$acc serial
      !CHECK: fir.call @_QMmod1Pcallee
      call callee(aa)
      !$acc end serial
    end subroutine
    !CHECK: }
    !CHECK: func.func private @_QMmod1Pcallee(!fir.ref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
end module