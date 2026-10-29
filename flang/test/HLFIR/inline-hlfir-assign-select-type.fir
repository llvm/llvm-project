// Test that hlfir.assign is not inlined when its LHS and RHS are overlapping
// parts of the same object accessed through views of different types.
// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// The assignment comes from the following source:
//   module m
//     type base
//       real :: v(4)
//     end type
//     type, extends(base) :: child
//       real :: extra(4)
//     end type
//   contains
//     subroutine shift(this)
//       class(base), intent(inout) :: this
//       select type (a => this)
//       type is (child)
//         a%v(2:4) = this%v(1:3)
//       end select
//     end subroutine
//   end module
// a%v is a%base%v, which overlaps this%v.

!base = !fir.type<_QMmTbase{v:!fir.array<4xf32>}>
!child = !fir.type<_QMmTchild{base:!base,extra:!fir.array<4xf32>}>
fir.type_info @_QMmTbase : !base
fir.type_info @_QMmTchild extends !base : !child
func.func @_QMmPshift(%arg0: !fir.class<!base> {fir.bindc_name = "this"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFshiftEthis") fortran_attrs<intent_inout> : (!fir.class<!base>, !fir.dscope) -> (!fir.class<!base>, !fir.class<!base>)
  fir.select_type %1#1 : !fir.class<!base> [#fir.type_is<!child>, ^bb1, unit, ^bb2]
^bb1:
  %2 = fir.box_addr %1#1 : (!fir.class<!base>) -> !fir.ref<!child>
  %3:2 = hlfir.declare %2 uniq_name("_QMmFshiftEa") : (!fir.ref<!child>) -> (!fir.ref<!child>, !fir.ref<!child>)
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %5 = fir.shape %c3 : (index) -> !fir.shape<1>
  %6 = hlfir.designate %1#0{"v"} <%4> (%c1:%c3:%c1)  shape %5 : (!fir.class<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %7 = hlfir.designate %3#0{"base"}   : (!fir.ref<!child>) -> !fir.ref<!base>
  %8 = hlfir.designate %7{"v"} <%4> (%c2:%c4:%c1)  shape %5 : (!fir.ref<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  hlfir.assign %6 to %8 : !fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>
  cf.br ^bb2
^bb2:
  return
}
// CHECK-LABEL: func.func @_QMmPshift(
// CHECK-NOT:     fir.do_loop
// CHECK:         hlfir.assign %{{.*}} to %{{.*}} : !fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>
// CHECK-NOT:     fir.do_loop
// CHECK:         return
