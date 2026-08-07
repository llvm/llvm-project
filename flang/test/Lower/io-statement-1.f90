! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! UNSUPPORTED: system-windows

 logical :: existsvar
 integer :: length
 real :: a(100)

  ! CHECK-LABEL: _QQmain
  ! CHECK: fir.call @_FortranAioBeginOpenUnit
  ! CHECK-DAG: fir.call @_FortranAioSetFile
  ! CHECK-DAG: fir.call @_FortranAioSetAccess
  ! CHECK: fir.call @_FortranAioEndIoStatement
  open(8, file="foo", access="sequential")

  ! CHECK: fir.call @_FortranAioBeginBackspace
  ! CHECK: fir.call @_FortranAioEndIoStatement
  backspace(8)

  ! CHECK: fir.call @_FortranAioBeginFlush
  ! CHECK: fir.call @_FortranAioEndIoStatement
  flush(8)

  ! CHECK: fir.call @_FortranAioBeginRewind
  ! CHECK: fir.call @_FortranAioEndIoStatement
  rewind(8)

  ! CHECK: fir.call @_FortranAioBeginEndfile
  ! CHECK: fir.call @_FortranAioEndIoStatement
  endfile(8)

  ! CHECK: fir.call @_FortranAioBeginWaitAll(%{{.*}}, %{{.*}}, %{{.*}})
  ! CHECK: fir.call @_FortranAioEndIoStatement
  wait(unit=8)

  ! CHECK: fir.call @_FortranAioBeginExternalListInput
  ! CHECK: fir.call @_FortranAioInputInteger
  ! CHECK: fir.call @_FortranAioInputReal32
  ! CHECK: fir.call @_FortranAioEndIoStatement
  read (8,*) i, f

  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.call @_FortranAioOutputInteger32
  ! CHECK: fir.call @_FortranAioOutputReal32
  ! CHECK: fir.call @_FortranAioEndIoStatement
  write (8,*) i, f

  ! CHECK: fir.call @_FortranAioBeginClose
  ! CHECK: fir.call @_FortranAioEndIoStatement
  close(8)

  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.call @_FortranAioOutputAscii
  ! CHECK: fir.call @_FortranAioEndIoStatement
  print *, "A literal string"

  ! CHECK: fir.call @_FortranAioBeginInquireUnit
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(4, EXIST=existsvar)

  ! CHECK: fir.call @_FortranAioBeginInquireFile
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(FILE="fail.f90", EXIST=existsvar)

  ! CHECK: fir.call @_FortranAioBeginInquireIoLength
  ! CHECK-COUNT-3: fir.call @_FortranAioOutputDescriptor
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire (iolength=length) existsvar, length, a
end

! CHECK-LABEL: internalnamelistio
subroutine internalNamelistIO()
  ! CHECK: %[[internal_var:.*]] = fir.alloca !fir.char<1,12> {bindc_name = "internal"
  ! CHECK: %[[internal_decl:.*]]:2 = hlfir.declare %[[internal_var]]
  character(12) :: internal
  integer :: x = 123
  namelist /nml/x
  ! CHECK: %[[internal_ptr:.*]] = fir.convert %[[internal_decl]]#0 : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: %[[cookie:.*]] = fir.call @_FortranAioBeginInternalListOutput(%[[internal_ptr]]
  ! CHECK: fir.call @_FortranAioOutputNamelist(%[[cookie]]
  ! CHECK: fir.call @_FortranAioEndIoStatement(%[[cookie]]
  write(internal,nml=nml)
end

! Tests the 4 basic inquire formats
! CHECK-LABEL: func @_QPinquire_test
subroutine inquire_test(ch, i, b)
  character(80) :: ch
  integer :: i
  logical :: b
  integer :: id_func

  ! CHARACTER
  ! CHECK: %[[sugar:.*]] = fir.call @_FortranAioBeginInquireUnit
  ! CHECK: fir.call @_FortranAioInquireCharacter(%[[sugar]], %c{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(88, name=ch)

  ! INTEGER
  ! CHECK: %[[oatmeal:.*]] = fir.call @_FortranAioBeginInquireUnit
  ! CHECK: fir.call @_FortranAioInquireInteger64(%[[oatmeal]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i64>, i32) -> i1
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(89, pos=i)

  ! LOGICAL
  ! CHECK: %[[snicker:.*]] = fir.call @_FortranAioBeginInquireUnit
  ! CHECK: fir.call @_FortranAioInquireLogical(%[[snicker]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i1>) -> i1
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(90, opened=b)

  ! PENDING with ID
  ! CHECK-DAG: %[[chip:.*]] = fir.call @_FortranAioBeginInquireUnit
  ! CHECK-DAG: fir.call @_QPid_func
  ! CHECK: fir.call @_FortranAioInquirePendingId(%[[chip]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32, !fir.ref<i1>) -> i1
  ! CHECK: fir.call @_FortranAioEndIoStatement
  inquire(91, id=id_func(), pending=b)
end subroutine inquire_test

! CHECK-LABEL: @_QPboz
subroutine boz
  ! CHECK: fir.call @_FortranAioOutputInteger8(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i8) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger16(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i16) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger128(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i128) -> i1
  print '(*(Z3))', 96_1, 96_2, 96_4, 96_8, 96_16

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  print '(I3,2Z44)', 40, 2**40_8, 2**40_8+1

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  print '(I3,2I44)', 40, 1099511627776,  1099511627777

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  print '(I3,2O44)', 40, 2**40_8, 2**40_8+1

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64) -> i1
  print '(I3,2B44)', 40, 2**40_8, 2**40_8+1
end
