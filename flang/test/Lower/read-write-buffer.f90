! RUN: bbc -emit-fir %s -o - | FileCheck %s
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
! CHECK: %[[lit:.*]] = fir.string_lit "compiler"(8) : !fir.char<1>
! CHECK: fir.has_value %[[lit]] : !fir.array<8x!fir.char<1>>
! CHECK: }
