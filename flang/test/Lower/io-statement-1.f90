! RUN: bbc %s -o - | FileCheck %s

 logical :: existsvar
 integer :: length
 real :: a(100)

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
  ! Note that 32 bit integers are output as 64 bits in the runtime API
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

  ! CHECK: call {{.*}}BeginInquireUnit
  ! CHECK: call {{.*}}EndIoStatement
  inquire(4, EXIST=existsvar)

  ! CHECK: call {{.*}}BeginInquireFile
  ! CHECK: call {{.*}}EndIoStatement
  inquire(FILE="fail.f90", EXIST=existsvar)

  ! CHECK: call {{.*}}BeginInquireIoLength
  ! CHECK: call {{.*}}EndIoStatement
  inquire (iolength=length) a
end

! Tests the 4 basic inquire formats
! CHECK-LABEL: func @_QPinquire_test
subroutine inquire_test(ch, i, b)
  character(80) :: ch
  integer :: i
  logical :: b
  integer :: id_func

  ! CHARACTER
  ! CHECK: %[[sugar:.*]] = fir.call {{.*}}BeginInquireUnit
  ! CHECK: call {{.*}}InquireCharacter(%[[sugar]], %c{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64) -> i1
  ! CHECK: call {{.*}}EndIoStatement
  inquire(88, name=ch)

  ! INTEGER
  ! CHECK: %[[oatmeal:.*]] = fir.call {{.*}}BeginInquireUnit
  ! CHECK: call @_FortranAioInquireInteger64(%[[oatmeal]], %c{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i64>, i32) -> i1
  ! CHECK: call {{.*}}EndIoStatement
  inquire(89, pos=i)

  ! LOGICAL
  ! CHECK: %[[snicker:.*]] = fir.call {{.*}}BeginInquireUnit
  ! CHECK: call @_FortranAioInquireLogical(%[[snicker]], %c{{.*}}, %[[b:.*]]) : (!fir.ref<i8>, i64, !fir.ref<i1>) -> i1
  ! CHECK: call {{.*}}EndIoStatement
  inquire(90, opened=b)

  ! PENDING with ID
  ! CHECK-DAG: %[[chip:.*]] = fir.call {{.*}}BeginInquireUnit
  ! CHECK-DAG: fir.call @_QPid_func
  ! CHECK: call @_FortranAioInquirePendingId(%[[chip]], %{{.*}}, %[[b]]) : (!fir.ref<i8>, i64, !fir.ref<i1>) -> i1
  ! CHECK: call {{.*}}EndIoStatement
  inquire(91, id=id_func(), pending=b)
end subroutine inquire_test
