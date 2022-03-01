! RUN: bbc %s -emit-fir -o - | FileCheck %s
! UNSUPPORTED: system-windows

 logical :: existsvar
 integer :: length
 real :: a(100)

  ! CHECK-LABEL: _QQmain
  ! CHECK: call {{.*}}BeginExternalListInput
  ! CHECK: call {{.*}}InputInteger
  ! CHECK: call {{.*}}InputReal32
  ! CHECK: call {{.*}}EndIoStatement
  read (8,*) i, f

  ! CHECK: call {{.*}}BeginExternalListOutput
  ! CHECK: call {{.*}}OutputInteger32
  ! CHECK: call {{.*}}OutputReal32
  ! CHECK: call {{.*}}EndIoStatement
  write (8,*) i, f

  ! CHECK: call {{.*}}BeginExternalListOutput
  ! CHECK: call {{.*}}OutputAscii
  ! CHECK: call {{.*}}EndIoStatement
  print *, "A literal string"
end

! CHECK-LABEL: @_QPboz
subroutine boz
  ! CHECK: fir.call @_FortranAioOutputInteger8(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i8) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger16(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i16) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger128(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i128) -> i1
  print '(*(Z3))', 96_1, 96_2, 96_4, 96_8, 96_16

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  print '(I3,2Z44)', 40, 2**40_8, 2**40_8+1

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  print '(I3,2I44)', 40, 1099511627776,  1099511627777

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  print '(I3,2O44)', 40, 2**40_8, 2**40_8+1

  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i64) -> i1
  print '(I3,2B44)', 40, 2**40_8, 2**40_8+1
end
