! RUN: bbc %s -o "-" -emit-hlfir | FileCheck %s
! Test lowering of calls to procedures with implicit interfaces using different
! calls with different argument types, one of which is character
subroutine s2
  integer i(3)
! CHECK:  %[[a0:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "i", uniq_name = "_QFs2Ei"}
! CHECK:  %[[decl:.*]]:2 = hlfir.declare %[[a0]]
  ! CHECK: fir.call @_QPsub2(%[[decl]]#0) {{.*}}: (!fir.ref<!fir.array<3xi32>>) -> ()
  call sub2(i)
! CHECK:  %[[a1:.*]] = fir.address_of(@_QQclX3031323334) : !fir.ref<!fir.char<1,5>>
! CHECK:  %[[decl_char:.*]]:2 = hlfir.declare %[[a1]]
! CHECK:  %[[expr:.*]] = hlfir.as_expr %[[decl_char]]#0
! CHECK:  %[[assoc:.*]]:3 = hlfir.associate %[[expr]] {{.*}} {adapt.valuebyref}
! CHECK:  %[[embox:.*]] = fir.emboxchar %[[assoc]]#0
! CHECK:  %[[func:.*]] = fir.address_of(@_QPsub2)
! CHECK:  %[[cast_func:.*]] = fir.convert %[[func]]
  ! CHECK: fir.call %[[cast_func]](%[[embox]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call sub2("01234")
end
