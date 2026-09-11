!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fenumeration-type -fsyntax-only -DSTEP=1 -J%t %s
!RUN: %flang_fc1 -fenumeration-type -fsyntax-only -DSTEP=2 -J%t %s
!RUN: not %flang_fc1 -fenumeration-type -fsyntax-only -DSTEP=3 -J%t %s 2>&1 | FileCheck --check-prefix=CHECK-PRIVTYPE %s
!RUN: not %flang_fc1 -fenumeration-type -fsyntax-only -DSTEP=4 -J%t %s 2>&1 | FileCheck --check-prefix=CHECK-LEAK %s
!RUN: %flang_fc1 -fenumeration-type -fsyntax-only -DSTEP=5 -J%t %s

! Enumerator accessibility must survive being written to a module file and read
! back: each using unit below is compiled in a SEPARATE invocation that reloads
! m*.mod through the module-file reader, which test_modfile.py (text-only
! comparison) cannot exercise.  See enumeration-type-mod.f90 (m7) for the
! generated-text expectations.

#if STEP == 1
! Producers, written to %t.

! 'private :: color' is an access-STATEMENT on the type name; it does NOT set
! the enumerator default (F2023 7.6.2p2), so red/green/blue are PUBLIC while the
! type color is PRIVATE.
module m7a
  private :: color
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! Module default private with the type made public: the enumerators take the
! module default (PRIVATE) while color is PUBLIC.
module m7b
  private
  public :: color
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! A USE-renamed enumeration type must be written to the module file under its
! in-scope (renamed) spelling so it resolves on readback.
module m7c
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

module m7d
  use m7c, only: hue => color
  type(hue) :: v
end module
#endif

#if STEP == 2
! A PUBLIC enumerator of a PRIVATE type stays usable through the module file,
! and a PUBLIC type with PRIVATE enumerators still exposes the type.
subroutine use_public
  use m7a, only: green
  use m7b, only: color
end subroutine
#endif

#if STEP == 3
! color is PRIVATE in m7a; reading m7a.mod must still reject it.
subroutine use_private_type
  !CHECK-PRIVTYPE: 'color' is PRIVATE in 'm7a'
  use m7a, only: color
end subroutine
#endif

#if STEP == 4
! red is PRIVATE in m7b; a private enumerator must not leak through m7b.mod.
subroutine use_private_enumerator
  !CHECK-LEAK: 'red' is PRIVATE in 'm7b'
  use m7b, only: red
end subroutine
#endif

#if STEP == 5
! Reading m7d.mod back must resolve the renamed type spelling 'hue' that was
! written to the module file, not the module-original name 'color'.
subroutine use_renamed
  use m7d, only: v
end subroutine
#endif
