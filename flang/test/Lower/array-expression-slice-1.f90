! RUN: bbc -emit-hlfir -fwrapv -o - --outline-intrinsics %s | FileCheck %s

! CHECK-LABEL: func @_QQmain() attributes {fir.bindc_name = "P"} {
! CHECK:         %[[a1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFEa1"}
! CHECK:         %[[a2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFEa2"}
! CHECK:         %[[a3:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFEa3"}
! CHECK:         %[[iv:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFEiv"}

! CHECK:         fir.do_loop
! CHECK:           fir.do_loop
! CHECK:             fir.call @fir.cos
! CHECK:             hlfir.assign
! CHECK:           fir.call @fir.sin
! CHECK:           hlfir.assign

! a2 = a1(4, 2:10:3)
! CHECK:         %[[slice1:.*]] = hlfir.designate %[[a1]]#0 (%c4{{.*}}, %c2{{.*}}:%c10{{.*}}:%c3{{.*}})
! CHECK:         hlfir.assign %[[slice1]] to %[[a2]]#0

! a3(1:10:4) = a2
! CHECK:         %[[slice2:.*]] = hlfir.designate %[[a3]]#0 (%c1{{.*}}:%c10{{.*}}:%c4{{.*}})
! CHECK:         hlfir.assign %[[a2]]#0 to %[[slice2]]

! a2 = a2(iv)
! CHECK:         %[[iv_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi64> {
! CHECK:           fir.load
! CHECK:           fir.convert
! CHECK:           hlfir.yield_element
! CHECK:         }
! CHECK:         %[[vec_slice:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[iv_val:.*]] = hlfir.apply %[[iv_expr]], %[[idx]]
! CHECK:           %[[a2_elem:.*]] = hlfir.designate %[[a2]]#0 (%[[iv_val]])
! CHECK:           %[[val:.*]] = fir.load %[[a2_elem]]
! CHECK:           hlfir.yield_element %[[val]]
! CHECK:         }
! CHECK:         hlfir.assign %[[vec_slice]] to %[[a2]]#0

program p
  real :: a1(10,10)
  real :: a2(3)
  real :: a3(10)
  integer iv(3)
  integer k

  k = 0
  do j = 1, 10
     do i = 1, 10
        k = k + 1
        a1(i,j) = cos(real(k))
     end do
     a3(j) = sin(real(k))
  end do

  a2 = a1(4, 2:10:3)

  if (a1(4,2) .ne. a2(1)) print *, "mismatch 1", a2(1), a1(4,2)
  if (a1(4,5) .ne. a2(2)) print *, "mismatch 2", a2(2), a1(4,5)
  if (a1(4,8) .ne. a2(3)) print *, "mismatch 3", a2(3), a1(4,8)

  a3(1:10:4) = a2

  if (a1(4,2) .ne. a3(1)) print *, "mismatch 4", a1(4,2), a3(1)
  if (a1(4,5) .ne. a3(5)) print *, "mismatch 5", a1(4,5), a3(5)
  if (a1(4,8) .ne. a3(9)) print *, "mismatch 6", a1(4,8), a3(9)

  iv = (/ 3, 1, 2 /)

  a2 = a2(iv)

  if (a1(4,2) .ne. a2(2)) print *, "mismatch 7", a1(4,2), a2(2)
  if (a1(4,5) .ne. a2(3)) print *, "mismatch 8", a1(4,5), a2(3)
  if (a1(4,8) .ne. a2(1)) print *, "mismatch 9", a1(4,8), a2(1)

end program p

! CHECK-LABEL: func @_QPsub(
! CHECK-SAME:               %[[arg:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[unbox:.*]]:2 = fir.unboxchar %[[arg]]
! CHECK:         %[[addr:.*]] = fir.convert %[[unbox]]#0
! CHECK:         %[[a:.*]]:2 = hlfir.declare %[[addr]](%{{.*}}) typeparams %{{.*}}
! CHECK:         %[[slice:.*]] = hlfir.designate %[[a]]#0 (%c1{{.*}}:%c5{{.*}}:%c2{{.*}})
! CHECK:         fir.call @_FortranAioOutputDescriptor
! CHECK:       }

! Slice operation on array of CHARACTER
subroutine sub(a)
  character :: a(10)
  print *, "a = ", a(1:5:2)
end subroutine sub
