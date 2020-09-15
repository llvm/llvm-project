! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPformatassign
function formatAssign()
    real :: pi
    integer :: label
    logical :: flag

    ! CHECK-DAG: %[[ONE:.*]] = constant 100 : i32
    ! CHECK-DAG: %[[TWO:.*]] = constant 200 : i32
    ! CHECK: %{{.*}} = select %{{.*}}, %[[ONE]], %[[TWO]] : i32
    if (flag) then
       assign 100 to label
    else
       assign 200 to label
    end if

    ! CHECK: fir.select %{{.*}} [100, ^bb[[BLK1:.*]], 200, ^bb[[BLK2:.*]], unit, ^bb[[BLK3:.*]]]
    ! CHECK: ^bb[[BLK1]]:
    ! CHECK: fir.address_of(@_QQcl
    ! CHECK: br ^bb[[END_BLOCK:.*]](
    ! CHECK: ^bb[[BLK2]]:
    ! CHECK: fir.address_of(@_QQcl
    ! CHECK: br ^bb[[END_BLOCK]](
    ! CHECK: ^bb[[BLK3]]:
    ! CHECK-NEXT: fir.unreachable
    ! CHECK: ^bb[[END_BLOCK]](
    ! CHECK: fir.call @{{.*}}BeginExternalFormattedOutput
    ! CHECK: fir.call @{{.*}}OutputAscii
    ! CHECK: fir.call @{{.*}}OutputReal32
    ! CHECK: fir.call @{{.*}}EndIoStatement
    pi = 3.141592653589
    write(*, label) "PI=", pi
 

100 format (A, F10.3)
200 format (A,E8.1)
300 format (A, E2.4)

    end function
