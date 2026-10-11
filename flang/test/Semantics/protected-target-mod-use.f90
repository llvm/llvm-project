! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/ok.f90 2>&1 | FileCheck --allow-empty --check-prefix=OK %s
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/ok.f90 2>&1 | FileCheck --check-prefix=PEDANTIC %s
! RUN: not %flang_fc1 -fsyntax-only -J%t %t/bad.f90 2>&1 | FileCheck %s

! PROTECTED_TARGET survives a module file, so its constraints apply to the
! entities of a module that is read back from its .mod file.

!--- m.f90
module protected_target_mod_use_m
  integer, target, protected :: px = 1
  integer, pointer, protected_target :: mp
  type :: t
    integer, pointer, protected_target :: c
  end type
 contains
  subroutine take_pt(d)
    integer, pointer, protected_target :: d
  end
  subroutine take_ptr(d)
    integer, pointer :: d
  end
  function get() result(r)
    integer, pointer, protected_target :: r
    r => px
  end
end

!--- ok.f90
subroutine ok
  use protected_target_mod_use_m
  integer, pointer, protected_target :: q
  type(t) :: a
  mp => px
  q => mp
  q => get()
  a = t(mp)
  call take_pt(mp)
  print *, mp, q, a%c
end

!--- bad.f90
subroutine bad
  use protected_target_mod_use_m
  integer, pointer :: w
  type(t) :: a
  mp = 2
  w => mp
  w => get()
  a%c = 3
  deallocate(mp)
 contains
  subroutine inner(q)
    integer, pointer, protected_target :: q
    call take_ptr(q)
  end
end

! OK-NOT: error
! OK-NOT: warning

! F'2028 C864 forbids "mp => px", although 8.5.16 NOTE 1 shows the equivalent
! as valid; it gets only the optional warning.
! PEDANTIC-NOT: error
! PEDANTIC: warning: Pointer target is not a definable variable [-Wpointer-to-undefinable]
! PEDANTIC: because: 'px' is protected in this scope
! PEDANTIC-NOT: error
! PEDANTIC-NOT: warning

! CHECK: error: Left-hand side of assignment is not definable
! CHECK: because: 'mp' has the PROTECTED_TARGET attribute
! CHECK: error: The target of PROTECTED_TARGET pointer 'mp' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
! CHECK: error: The target of PROTECTED_TARGET pointer 'r' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
! CHECK: error: Left-hand side of assignment is not definable
! CHECK: because: 'c' has the PROTECTED_TARGET attribute
! CHECK: error: Object in DEALLOCATE statement is not deallocatable
! CHECK: because: 'mp' has the PROTECTED_TARGET attribute
! CHECK: error: The target of PROTECTED_TARGET pointer 'q' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
