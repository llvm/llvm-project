! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1a(
! CHECK-SAME:    %[[a:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[c:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}) {
! CHECK:         %[[a_decl:.*]]:2 = hlfir.declare %[[a]]
! CHECK:         %[[b_decl:.*]]:2 = hlfir.declare %[[b]]
! CHECK:         %[[c_decl:.*]]:2 = hlfir.declare %[[c]]
! CHECK:         %[[c_slice:.*]] = hlfir.designate %[[c_decl]]#0 (%c1{{.*}}:%c20{{.*}}:%c2{{.*}})
! CHECK:         %[[c_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[c_elem:.*]] = hlfir.designate %[[c_slice]] (%[[idx]])
! CHECK:           %[[val:.*]] = fir.load %[[c_elem]]
! CHECK:           %[[cast:.*]] = fir.convert %[[val]]
! CHECK:           hlfir.yield_element %[[cast]]
! CHECK:         }
! CHECK:         %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[c_val:.*]] = hlfir.apply %[[c_expr]], %[[idx]]
! CHECK:           %[[b_elem:.*]] = hlfir.designate %[[b_decl]]#0 (%[[c_val]])
! CHECK:           %[[b_val:.*]] = fir.load %[[b_elem]]
! CHECK:           hlfir.yield_element %[[b_val]]
! CHECK:         }
! CHECK:         hlfir.assign %[[res]] to %[[a_decl]]#0

subroutine test1a(a,b,c)
  integer :: a(10), b(10), c(20)

  a = b(c(1:20:2))
end subroutine test1a

! CHECK-LABEL: func @_QPtest1b(
! CHECK-SAME:      %[[a:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[c:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}) {
! CHECK:         %[[a_decl:.*]]:2 = hlfir.declare %[[a]]
! CHECK:         %[[b_decl:.*]]:2 = hlfir.declare %[[b]]
! CHECK:         %[[c_decl:.*]]:2 = hlfir.declare %[[c]]
! CHECK:         hlfir.region_assign {
! CHECK:           hlfir.yield %[[a_decl]]#0
! CHECK:         } to {
! CHECK:           %[[c_slice:.*]] = hlfir.designate %[[c_decl]]#0 (%c1{{.*}}:%c20{{.*}}:%c2{{.*}})
! CHECK:           %[[c_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:           ^bb0(%[[idx:.*]]: index):
! CHECK:             %[[c_elem:.*]] = hlfir.designate %[[c_slice]] (%[[idx]])
! CHECK:             %[[val:.*]] = fir.load %[[c_elem]]
! CHECK:             %[[cast:.*]] = fir.convert %[[val]]
! CHECK:             hlfir.yield_element %[[cast]]
! CHECK:           }
! CHECK:           hlfir.elemental_addr %{{.*}} unordered : !fir.shape<1> {
! CHECK:           ^bb0(%[[idx:.*]]: index):
! CHECK:             %[[c_val:.*]] = hlfir.apply %[[c_expr]], %[[idx]]
! CHECK:             %[[b_elem:.*]] = hlfir.designate %[[b_decl]]#0 (%[[c_val]])
! CHECK:             hlfir.yield %[[b_elem]]
! CHECK:           }
! CHECK:         }

subroutine test1b(a,b,c)
  integer :: a(10), b(10), c(20)

  b(c(1:20:2)) = a
end subroutine test1b

! CHECK-LABEL: func @_QPtest2a(
! CHECK-SAME:     %[[a:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[c:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[d:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         %[[a_decl:.*]]:2 = hlfir.declare %[[a]]
! CHECK:         %[[b_decl:.*]]:2 = hlfir.declare %[[b]]
! CHECK:         %[[c_decl:.*]]:2 = hlfir.declare %[[c]]
! CHECK:         %[[d_decl:.*]]:2 = hlfir.declare %[[d]]
! CHECK:         %[[d_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:           fir.load
! CHECK:           fir.convert
! CHECK:           hlfir.yield_element
! CHECK:         }
! CHECK:         %[[c_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[d_val:.*]] = hlfir.apply %[[d_expr]], %[[idx]]
! CHECK:           %[[c_elem:.*]] = hlfir.designate %[[c_decl]]#0 (%[[d_val]])
! CHECK:           %[[c_val:.*]] = fir.load %[[c_elem]]
! CHECK:           hlfir.yield_element %[[c_val]]
! CHECK:         }
! CHECK:         %[[c_expr_cast:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:           fir.convert
! CHECK:           hlfir.yield_element
! CHECK:         }
! CHECK:         %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[c_val:.*]] = hlfir.apply %[[c_expr_cast]], %[[idx]]
! CHECK:           %[[b_elem:.*]] = hlfir.designate %[[b_decl]]#0 (%[[c_val]])
! CHECK:           %[[b_val:.*]] = fir.load %[[b_elem]]
! CHECK:           hlfir.yield_element %[[b_val]]
! CHECK:         }
! CHECK:         hlfir.assign %[[res]] to %[[a_decl]]#0

subroutine test2a(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  a = b(c(d))
end subroutine test2a

! CHECK-LABEL: func @_QPtest2b(
! CHECK-SAME:      %[[a:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[c:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[d:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         %[[a_decl:.*]]:2 = hlfir.declare %[[a]]
! CHECK:         %[[b_decl:.*]]:2 = hlfir.declare %[[b]]
! CHECK:         %[[c_decl:.*]]:2 = hlfir.declare %[[c]]
! CHECK:         %[[d_decl:.*]]:2 = hlfir.declare %[[d]]
! CHECK:         hlfir.region_assign {
! CHECK:           hlfir.yield %[[a_decl]]#0
! CHECK:         } to {
! CHECK:           %[[d_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:             fir.load
! CHECK:             fir.convert
! CHECK:             hlfir.yield_element
! CHECK:           }
! CHECK:           %[[c_expr:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[idx:.*]]: index):
! CHECK:             %[[d_val:.*]] = hlfir.apply %[[d_expr]], %[[idx]]
! CHECK:             %[[c_elem:.*]] = hlfir.designate %[[c_decl]]#0 (%[[d_val]])
! CHECK:             %[[c_val:.*]] = fir.load %[[c_elem]]
! CHECK:             hlfir.yield_element %[[c_val]]
! CHECK:           }
! CHECK:           %[[c_expr_cast:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi64> {
! CHECK:             fir.convert
! CHECK:             hlfir.yield_element
! CHECK:           }
! CHECK:           hlfir.elemental_addr %{{.*}} unordered : !fir.shape<1> {
! CHECK:           ^bb0(%[[idx:.*]]: index):
! CHECK:             %[[c_val:.*]] = hlfir.apply %[[c_expr_cast]], %[[idx]]
! CHECK:             %[[b_elem:.*]] = hlfir.designate %[[b_decl]]#0 (%[[c_val]])
! CHECK:             hlfir.yield %[[b_elem]]
! CHECK:           }
! CHECK:         }

subroutine test2b(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  b(c(d)) = a
end subroutine test2b

! CHECK-LABEL: func @_QPtest1c(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         return
! CHECK:       }
subroutine test1c(a,b,c,d)
  integer :: a(10), b(10), d(10), c(20)

  ! flang: parser FAIL (final position)
  !a = b(d(c(1:20:2))
end subroutine test1c
