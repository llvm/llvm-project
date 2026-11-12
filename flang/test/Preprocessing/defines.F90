! RUN: %flang -E -DFOO=1 -DBAR=2 %s | FileCheck %s

! CHECK: integer :: a = 1
  integer :: a = FOO
! CHECK: integer :: b = 2
  integer :: b = BAR

end program
