! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/s.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/pt-mod.f90
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/pt-pass.f90
! RUN: not %flang_fc1 -fsyntax-only -module-dir %t %t/pt-fail.f90 2>&1 | FileCheck --check-prefix=PT %s

! When a module and its submodule are compiled in separate
! invocations (so that the submodule's parent module scope is read back from
! the .mod file), the "missing MODULE prefix" portability warning for the
! submodule's subprogram must still be emitted.

!--- m.f90
module modfile87m
  interface
    module subroutine inside_one()
    end subroutine
  end interface
end module

!--- s.f90
submodule (modfile87m) modfile87s
contains
  subroutine inside_one()
  end subroutine
end submodule

!CHECK: portability: Subprogram 'inside_one' in this submodule is missing the MODULE prefix

!--- pt-mod.f90
module pt_mod
  implicit none
  integer, target :: x
  integer, pointer, protected_target :: p
contains
  subroutine takes_pt(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  subroutine takes_plain(a)
    integer, pointer, intent(in) :: a
  end subroutine
  function pt_result() result(r)
    integer, pointer, protected_target :: r
    r => x
  end function
end module

!--- pt-pass.f90
subroutine pass_case()
  use pt_mod
  integer, pointer, protected_target :: r
  call takes_pt(p)
  call takes_pt(pt_result())
  r => pt_result()
end subroutine

!--- pt-fail.f90
subroutine fail_case()
  use pt_mod
  integer, pointer :: q
  q => p
  q => pt_result()
  call takes_plain(p)
end subroutine

!PT: error: Pointer assignment target 'p' has the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
!PT: error: Pointer assignment target is function 'pt_result' result with the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
!PT: error: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
