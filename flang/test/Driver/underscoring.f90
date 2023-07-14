! Test the -funderscoring flag

! RUN: %flang_fc1 -S %s -o - 2>&1 | FileCheck %s --check-prefix=UNDERSCORING
! RUN: %flang_fc1 -S -fno-underscoring %s -o - 2>&1 | FileCheck %s --check-prefix=NO-UNDERSCORING

subroutine test()
  common /comblk/ a, b
  external :: ext_sub
  call ext_sub()
end

! UNDERSCORING: test_
! UNDERSCORING-NOT: {{test:$}}
! UNDERSCORING: ext_sub_
! UNDERSCORING-NOT: {{ext_sub[^_]*$}}
! UNDERSCORING: comblk_
! UNDERSCORING-NOT: comblk,

! NO-UNDERSCORING-NOT: test_
! NO-UNDERSCORING: test:
! NO-UNDERSCORING-NOT: ext_sub_
! NO-UNDERSCORING: {{ext_sub[^_]*$}}
! NO-UNDERSCORING-NOT: comblk_
! NO-UNDERSCORING: comblk,
