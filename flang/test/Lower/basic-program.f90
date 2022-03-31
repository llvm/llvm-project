! RUN: bbc %s --pft-test | FileCheck %s
! RUN: bbc %s -o "-" -emit-fir | FileCheck %s --check-prefix=FIR

program basic
end program

! CHECK: 1 Program basic
! CHECK:   1 EndProgramStmt: end program
! CHECK: End Program basic

! FIR-LABEL: func @_QQmain() {
! FIR:         return
! FIR:       }
