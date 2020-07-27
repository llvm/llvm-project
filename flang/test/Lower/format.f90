! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPformatassign
function formatAssign()
    real :: pi
    integer :: label
    logical :: flag

    ! CHECK: select
    if (flag) then
       assign 100 to label
    else
       assign 200 to label
    end if

    ! CHECK: fir.select {{.*\[100, \^bb[0-9]+, 200, \^bb[0-9]+, unit, \^bb[0-9]+\]}}
    ! CHECK-LABEL: ^bb{{[0-9]+}}:
    ! CHECK: fir.address_of
    ! CHECK: br [[END_BLOCK:\^bb[0-9]+]]{{(.*)}}
    ! CHECK-LABEL: ^bb{{[0-9]+}}: //
    ! CHECK: fir.address_of
    ! CHECK: br [[END_BLOCK]]
    ! CHECK-LABEL: ^bb{{[0-9]+}}: //
    ! CHECK: fir.address_of
    ! CHECK: br [[END_BLOCK]]
    ! CHECK-LABEL: ^bb{{[0-9]+(.*)}}: //
    ! CHECK: call{{.*}}BeginExternalFormattedOutput
    ! CHECK-DAG: call{{.*}}OutputAscii
    ! CHECK-DAG: call{{.*}}OutputReal32
    ! CHECK: call{{.*}}EndIoStatement
    pi = 3.141592653589
    write(*, label) "PI=", pi
 

100 format (A, F10.3)
200 format (A,E8.1)
300 format (A, E2.4)

    end function
