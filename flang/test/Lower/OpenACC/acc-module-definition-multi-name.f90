! This file is compiled as a prerequisite by acc-routine-module-multi-name.f90
! to generate mod_multi_name.mod.

! RUN: rm -fr %t && mkdir -p %t && cd %t
! RUN: bbc -fopenacc -emit-fir %s
! RUN: cat mod_multi_name.mod | FileCheck %s

!CHECK-LABEL: module mod_multi_name
module mod_multi_name

  ! Two-name form: seq1 and seq2 should both get !$acc routine seq in the mod.
  !$acc routine(seq1, seq2) seq

  ! Three-name form: gang1, gang2, gang3 should all get !$acc routine gang.
  !$acc routine(gang1, gang2, gang3) gang

contains

  !CHECK-LABEL: subroutine seq1
  subroutine seq1()
  !CHECK: !$acc routine seq
  end subroutine

  !CHECK-LABEL: subroutine seq2
  subroutine seq2()
  !CHECK: !$acc routine seq
  end subroutine

  !CHECK-LABEL: subroutine gang1
  subroutine gang1()
  !CHECK: !$acc routine gang
  end subroutine

  !CHECK-LABEL: subroutine gang2
  subroutine gang2()
  !CHECK: !$acc routine gang
  end subroutine

  !CHECK-LABEL: subroutine gang3
  subroutine gang3()
  !CHECK: !$acc routine gang
  end subroutine

end module
