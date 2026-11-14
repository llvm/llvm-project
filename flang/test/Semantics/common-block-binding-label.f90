! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! RUN: %python %S/test_errors.py %s %flang_fc1 -Wcommon-block-binding-label -Werror
! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | count 0
! RUN: %flang_fc1 -fsyntax-only -pedantic -Wno-common-block-binding-label %s 2>&1 | count 0

! A default label is the lower-case COMMON name.  All three appearances
! share storage, but using different Fortran names is an extension.
subroutine default_label
  integer :: x
  common /Shared/ x
  bind(c) :: /Shared/
end subroutine

subroutine explicit_label
  integer :: x
  !PORTABILITY: Sharing binding label 'shared' between COMMON blocks with different names is not standard [-Wcommon-block-binding-label]
  common /other/ x
  bind(c, name="shared") :: /other/
end subroutine

subroutine third_label
  integer :: x
  !PORTABILITY: Sharing binding label 'shared' between COMMON blocks with different names is not standard [-Wcommon-block-binding-label]
  common /third/ x
  bind(c, name="shared") :: /third/
end subroutine

! Same-named COMMON blocks do not require the extension.
subroutine same_name1
  integer :: x
  common /same/ x
  bind(c, name="same_label") :: /same/
end subroutine

subroutine same_name2
  integer :: x
  common /same/ x
  bind(c, name="same_label") :: /same/
end subroutine

! Empty and all-blank NAME= values leave these same-named blocks unbound.
subroutine blank_label1
  integer :: x
  common /blank_label/ x
  bind(c, name="") :: /blank_label/
end subroutine

subroutine blank_label2
  integer :: x
  common /blank_label/ x
  bind(c, name="   ") :: /blank_label/
end subroutine
