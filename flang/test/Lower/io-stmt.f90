! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain
! CHECK: call {{.*}}BeginOpenUnit
! CHECK-DAG: call {{.*}}SetFile
! CHECK-DAG: call {{.*}}SetAccess
! CHECK: call {{.*}}EndIoStatement

  open(8, file="foo", access="sequential")

! CHECK: call {{.*}}BeginBackspace
! CHECK: call {{.*}}EndIoStatement
  backspace(8)
  
! CHECK: call {{.*}}BeginFlush
! CHECK: call {{.*}}EndIoStatement
  flush(8)
  
! CHECK: call {{.*}}BeginRewind
! CHECK: call {{.*}}EndIoStatement
  rewind(8)

! CHECK: call {{.*}}BeginEndfile
! CHECK: call {{.*}}EndIoStatement
  endfile(8)

! CHECK: call {{.*}}BeginWaitAll
! CHECK: call {{.*}}EndIoStatement
  wait(unit=8)
  
! CHECK: call {{.*}}BeginExternalListInput
! CHECK: call {{.*}}InputInteger
! CHECK: call {{.*}}InputReal32
! CHECK: call {{.*}}EndIoStatement
  read (8,*) i, f

! CHECK: call {{.*}}BeginExternalListOutput
! 32 bit integers are output as 64 bits in the runtime API  
! CHECK: call {{.*}}OutputInteger64
! CHECK: call {{.*}}OutputReal32
! CHECK: call {{.*}}EndIoStatement
  write (8,*) i, f

! CHECK: call {{.*}}BeginClose
! CHECK: call {{.*}}EndIoStatement
  close(8)

! CHECK: call {{.*}}BeginExternalListOutput
! CHECK: call {{.*}}OutputAscii
! CHECK: call {{.*}}EndIoStatement
  print *, "A literal string"
end
