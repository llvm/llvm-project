! RUN: %flang -Wknown-bad-implicit-interface %s -c 2>&1 | FileCheck %s --check-prefix=WARN
! RUN: %flang -pedantic -Wno-known-bad-implicit-interface %s -c 2>&1 | FileCheck %s --allow-empty
! RUN: not %flang -WKnownBadImplicitInterface %s -c 2>&1 | FileCheck %s --check-prefix=ERROR1
! RUN: not %flang -WKnown-Bad-Implicit-Interface %s -c 2>&1 | FileCheck %s --check-prefix=ERROR2

! ERROR1: error: Unknown diagnostic option: -WKnownBadImplicitInterface
! ERROR2: error: Unknown diagnostic option: -WKnown-Bad-Implicit-Interface

program disable_diagnostic
  ! CHECK-NOT: warning
  ! WARN: warning: If the procedure's interface were explicit, this reference would be in error
  call sub(1.)
  ! WARN: warning: If the procedure's interface were explicit, this reference would be in error
  call sub(1)
end program disable_diagnostic

subroutine sub()
end subroutine sub
