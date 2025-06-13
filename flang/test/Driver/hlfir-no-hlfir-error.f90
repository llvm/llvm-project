! Test that -flang-experimental-hlfir and -flang-deprecated-no-hlfir
! options cannot be both used.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: not %flang -flang-experimental-hlfir -flang-deprecated-no-hlfir %s 2>&1 | FileCheck %s

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 -emit-llvm -flang-experimental-hlfir -flang-deprecated-no-hlfir %s 2>&1 | FileCheck %s

! CHECK:error: Options '-flang-experimental-hlfir' and '-flang-deprecated-no-hlfir' cannot be both specified

end
