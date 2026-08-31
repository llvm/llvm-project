! RUN: %flang_fc1 -fimplicit-module-prefix -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! The enabled extension silently repairs a likely omitted MODULE prefix by
! binding the definition to the separate module procedure interface.
module m
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module m

submodule(m) sm
contains
  subroutine implementation
  end subroutine implementation
end submodule sm

! CHECK: Module scope: m
! CHECK: implementation, MODULE, PUBLIC (Subroutine): Subprogram isInterface ()
! CHECK: Module scope: sm
! CHECK: implementation, MODULE, PUBLIC (Subroutine): Subprogram () moduleInterface: implementation, MODULE, PUBLIC (Subroutine): Subprogram isInterface ()
