! RUN: %flang_fc1 -fimplicit-module-prefix -emit-hlfir %s -o - | FileCheck %s

module alpha
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module alpha

submodule(alpha) beta
end submodule beta

submodule(alpha:beta) gamma
contains
  ! CHECK-LABEL: func @_QMalphaPimplementation()
  subroutine implementation
  end subroutine implementation
end submodule gamma

! CHECK-LABEL: func @_QQmain()
program main
  use alpha
  ! CHECK: fir.call @_QMalphaPimplementation() {{.*}}
  call implementation
end program main
