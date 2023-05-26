! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Test checks whether the text of the format statement is hashconed into a
! global similar to a CHARACTER literal and then referenced.

program other
  write(10, 1008)
  ! CHECK:  fir.address_of(@{{.*}}) :
1008 format('ok')
end
! CHECK-LABEL: fir.global internal @_QQcl.28276F6B2729 constant
! CHECK: %[[lit:.*]] = fir.string_lit "('ok')"(6) : !fir.char<1,6>
! CHECK: fir.has_value %[[lit]] : !fir.char<1,6>
! CHECK: }
