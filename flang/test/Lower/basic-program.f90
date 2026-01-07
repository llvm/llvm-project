! RUN: bbc %s --pft-test | FileCheck %s
! RUN: bbc %s -o "-" -emit-fir | FileCheck %s --check-prefix=FIR

program basic
end program

! CHECK: 1 Program BASIC
! CHECK:   1 EndProgramStmt: end program
! CHECK: End Program BASIC

! FIR-LABEL: func @_QQmain() attributes {fir.bindc_name = "BASIC"} {
! FIR:         return
! FIR:       }
