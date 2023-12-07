! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPformatassign
subroutine formatAssign(flag1, flag2, flag3)
    real :: pi
    integer :: label
    logical :: flag1, flag2, flag3

    ! CHECK-DAG: %[[ONE:.*]] = arith.constant 100 : i32
    ! CHECK-DAG: %[[TWO:.*]] = arith.constant 200 : i32
    if (flag1) then
       assign 100 to label
    else
       assign 200 to label
    end if

    ! CHECK: cond_br %{{.*}}, ^bb[[BLK1:.*]], ^bb[[BLK2:.*]]
    ! CHECK: ^bb[[BLK1]]:
    ! CHECK: fir.store %[[ONE]]
    ! CHECK: br ^bb[[END_BLOCK:.*]]
    ! CHECK: ^bb[[BLK2]]:
    ! CHECK: fir.store %[[TWO]]
    ! CHECK: br ^bb[[END_BLOCK]]
    ! CHECK: ^bb[[END_BLOCK]]
    ! CHECK: fir.call @{{.*}}ReportFatalUserError
    ! CHECK: fir.unreachable
    ! CHECK: fir.call @{{.*}}BeginExternalFormattedOutput
    ! CHECK: fir.call @{{.*}}OutputAscii
    ! CHECK: fir.call @{{.*}}OutputReal32
    ! CHECK: fir.call @{{.*}}EndIoStatement
    pi = 3.141592653589
    write(*, label) " PI=", pi
    ! CHECK: fir.call @{{.*}}ReportFatalUserError
    ! CHECK: fir.unreachable
    ! CHECK: fir.call @{{.*}}BeginExternalFormattedOutput
    ! CHECK: fir.call @{{.*}}OutputAscii
    ! CHECK: fir.call @{{.*}}OutputReal32
    ! CHECK: fir.call @{{.*}}EndIoStatement
    if (flag2) write(*, label) "2PI=", 2*pi
    if (flag1 .and. flag2 .and. flag3) then
       assign 100 to label
    else
       assign 200 to label
    end if
    if (flag3) then
      ! CHECK: fir.call @{{.*}}ReportFatalUserError
      ! CHECK: fir.unreachable
      ! CHECK: fir.call @{{.*}}BeginExternalFormattedOutput
      ! CHECK: fir.call @{{.*}}OutputAscii
      ! CHECK: fir.call @{{.*}}OutputReal32
      ! CHECK: fir.call @{{.*}}EndIoStatement
      write(*, label) "3PI=", 3*pi
    endif

100 format (A, F10.3)
200 format (A,E8.1)
300 format (A, E4.2)

end subroutine

! CHECK-LABEL: func @_QQmain
  call formatAssign(.true., .true., .true.)
  print*
  call formatAssign(.true., .false., .true.)
end
