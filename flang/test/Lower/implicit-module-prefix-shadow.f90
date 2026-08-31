! RUN: %flang_fc1 -Wno-missing-module-prefix -emit-hlfir %s -o - | FileCheck %s

! A local procedure in beta hides alpha's interface only in beta and its
! descendants. The sibling submodule may still implement alpha's public
! second procedure.
module alpha
  interface
    module subroutine second
    end subroutine second
    module subroutine third
    end subroutine third
  end interface
end module alpha

submodule(alpha) beta
contains
  ! CHECK-LABEL: func @_QMalphaSbetaPsecond()
  subroutine second
  end subroutine second
end submodule beta

submodule(alpha:beta) nested_gamma
contains
  ! CHECK-LABEL: func @_QMalphaPthird()
  module subroutine third
  end subroutine third
end submodule nested_gamma

submodule(alpha) sibling_gamma
contains
  ! CHECK-LABEL: func @_QMalphaPsecond()
  module subroutine second
  end subroutine second
end submodule sibling_gamma

! CHECK-LABEL: func @_QQmain()
program main
  use alpha
  ! CHECK: fir.call @_QMalphaPsecond() {{.*}}
  call second
  ! CHECK: fir.call @_QMalphaPthird() {{.*}}
  call third
end program main
