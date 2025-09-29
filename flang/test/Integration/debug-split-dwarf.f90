! REQUIRES: x86-registered-target

! Testing to ensure that setting only -split-dwarf-file allows to place
! .dwo sections into regular output object.
!  RUN: %flang_fc1 -debug-info-kind=standalone -triple x86_64-unknown-linux \
!  RUN:   -split-dwarf-file %t.o -emit-obj -o %t.o %s
!  RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=DWO %s

! Testing to ensure that setting both -split-dwarf-file and -split-dwarf-output
! does not place .dwo sections into regular output object but in a separate
! file.
!  RUN: %flang_fc1 -debug-info-kind=standalone -triple x86_64-unknown-linux \
!  RUN:   -split-dwarf-file %t.dwo -split-dwarf-output %t.dwo -emit-obj -o %t.o %s
!  RUN: llvm-readobj -S %t.dwo | FileCheck --check-prefix=DWO %s
!  RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=SPLIT %s

!  DWO: .dwo
!  SPLIT-NOT: .dwo

program test
end program test
