! RUN: bbc %s -o "-" -emit-fir -hlfir=false | FileCheck %s
! Test lowering of calls to procedures with implicit interfaces using different
! calls with different argument types, one of which is character
subroutine s2
  integer i(3)
! CHECK:  %[[a0:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "i", uniq_name = "_QFs2Ei"}
  ! CHECK: fir.call @_QPsub2(%[[a0]]) {{.*}}: (!fir.ref<!fir.array<3xi32>>) -> ()
  call sub2(i)
! CHECK:  %[[a1:.*]] = fir.address_of(@_QQclX3031323334) : !fir.ref<!fir.char<1,5>>
! CHECK:  %[[a2:.*]] = fir.convert %[[a1]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.array<3xi32>>
  ! CHECK: fir.call @_QPsub2(%[[a2]]) {{.*}}: (!fir.ref<!fir.array<3xi32>>) -> ()
  call sub2("01234")
end
