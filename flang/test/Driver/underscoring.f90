! Test the -funderscoring flag

! RUN: %flang_fc1 -S %s -o - 2>&1 | FileCheck %s --check-prefix=UNDERSCORING
! RUN: %flang_fc1 -S -fno-underscoring %s -o - 2>&1 | FileCheck %s --check-prefix=NO-UNDERSCORING

subroutine test()
  common /comblk/ a, b
  external :: ext_sub
  call ext_sub()
end

! UNDERSCORING: test_
! UNDERSCORING: ext_sub_
! UNDERSCORING: comblk_

! NO-UNDERSCORING-NOT: test_
! NO-UNDERSCORING-NOT: _QPtest
! NO-UNDERSCORING: test
! NO-UNDERSCORING-NOT: ext_sub_
! NO-UNDERSCORING-NOT: _QPext_sub
! NO-UNDERSCORING: ext_sub
! NO-UNDERSCORING-NOT: comblk_
! NO-UNDERSCORING-NOT: _QBcomblk
! NO-UNDERSCORING: comblk
