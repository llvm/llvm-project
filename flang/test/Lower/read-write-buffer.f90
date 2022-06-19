! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test that we are passing the correct length when using character array as
! Format (Fortran 2018 12.6.2.2 point 3)
! CHECK-LABEL: func @_QPtest_array_format
subroutine test_array_format
  ! CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
  ! CHECK-DAG: %[[c10:.*]] = arith.constant 10 : index
  ! CHECK-DAG: %[[mem:.*]] = fir.alloca !fir.array<2x!fir.char<1,10>>
  character(10) :: array(2)
  array(1) ="(15HThis i"
  array(2) ="s a test.)"
  ! CHECK-DAG: %[[fmtLen:.*]] = arith.muli %[[c10]], %[[c2]] : index
  ! CHECK-DAG: %[[scalarFmt:.*]] = fir.convert %[[mem]] : (!fir.ref<!fir.array<2x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK-DAG: %[[fmtArg:.*]] = fir.convert %[[scalarFmt]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[fmtLenArg:.*]] = fir.convert %[[fmtLen]] : (index) -> i64 
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedOutput(%[[fmtArg]], %[[fmtLenArg]], {{.*}}) 
  write(*, array) 
end subroutine

! A test to check the buffer and it's length.
! CHECK-LABEL: @_QPsome
subroutine some()
  character(LEN=255):: buffer
  character(LEN=255):: greeting
10 format (A255)
  ! CHECK:  fir.address_of(@_QQcl.636F6D70696C6572) :
  write (buffer, 10) "compiler"
  read (buffer, 10) greeting
end
! CHECK-LABEL: fir.global linkonce @_QQcl.636F6D70696C6572
! CHECK: %[[lit:.*]] = fir.string_lit "compiler"(8) : !fir.char<1,8>
! CHECK: fir.has_value %[[lit]] : !fir.char<1,8>
! CHECK: }
