// Test code generation of hlfir.region_assign with user defined
// assignment implying finalization of the RHS.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

//! module types
//!   type ud_assign
//!      integer, pointer :: x
//!      contains
//!        procedure :: assign
//!        generic :: assignment(=) => assign
//!        final :: finalize
//!   end type ud_assign
//!   type ud_pure_assign
//!      integer, pointer :: x
//!      contains
//!        procedure :: pure_assign
//!        generic :: assignment(=) => pure_assign
//!        final :: pure_finalize
//!   end type ud_pure_assign
//! contains
//!   elemental impure subroutine assign(lhs, rhs)
//!     class(ud_assign), intent(inout) :: lhs
//!     class(ud_assign), intent(in) :: rhs
//!   end subroutine assign
//!   elemental impure subroutine finalize(this)
//!     type(ud_assign), intent(inout) :: this
//!   end subroutine finalize
//!   elemental pure subroutine pure_assign(lhs, rhs)
//!     class(ud_pure_assign), intent(inout) :: lhs
//!     class(ud_pure_assign), intent(in) :: rhs
//!   end subroutine pure_assign
//!   elemental pure subroutine pure_finalize(this)
//!     type(ud_pure_assign), intent(inout) :: this
//!   end subroutine pure_finalize
//! end module types

//! subroutine test1()
//!   use types
//!   interface
//!      function new_obj()
//!        use types
//!        type(ud_assign) :: new_obj
//!      end function new_obj
//!   end interface
//!   type(ud_assign), save :: x
//!   x = new_obj()
//! end subroutine test1
func.func @_QPtest1() {
  %0 = fir.alloca !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}> {bindc_name = ".result"}
  %1 = fir.address_of(@_QFtest1Ex) : !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>)
  hlfir.region_assign {
    %3 = fir.call @_QPnew_obj() fastmath<contract> : () -> !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>
    fir.save_result %3 to %0 : !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %4:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>)
    hlfir.yield %4#0 : !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>> cleanup {
      %5 = fir.embox %0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
      %6 = fir.convert %5 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<none>
      %7 = fir.call @_FortranADestroy(%6) fastmath<contract> : (!fir.box<none>) -> none
    }
  } to {
    hlfir.yield %2#0 : !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
  } user_defined_assign  (%arg0: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) to (%arg1: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) {
    %3 = fir.embox %arg1 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %4 = fir.convert %3 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %5 = fir.embox %arg0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %6 = fir.convert %5 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    fir.call @_QMtypesPassign(%4, %6) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}> {bindc_name = ".result"}
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFtest1Ex) : !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>)
// CHECK:           %[[VAL_3:.*]] = fir.call @_QPnew_obj() fastmath<contract> : () -> !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>
// CHECK:           fir.save_result %[[VAL_3]] to %[[VAL_0]] : !fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.as_expr %[[VAL_4]]#0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !hlfir.expr<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {uniq_name = ".tmp.assign"} : (!hlfir.expr<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, i1)
// CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_2]]#0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_6]]#0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           fir.call @_QMtypesPassign(%[[VAL_8]], %[[VAL_10]]) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
// CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, i1
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<none>
// CHECK:           %[[VAL_13:.*]] = fir.call @_FortranADestroy(%[[VAL_12]]) fastmath<contract> : (!fir.box<none>) -> none
// CHECK:           return
// CHECK:         }

//! subroutine test2()
//!   use types
//!   interface
//!      function new_obja()
//!        use types
//!        type(ud_assign) :: new_obja(2)
//!      end function new_obja
//!   end interface
//!   type(ud_assign), save :: xa(2)
//!   xa = new_obja()
//! end subroutine test2
func.func @_QPtest2() {
  %c2 = arith.constant 2 : index
  %0 = fir.alloca !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>> {bindc_name = ".result"}
  %1 = fir.address_of(@_QFtest2Exa) : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
  %2 = fir.shape %c2 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest2Exa"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
  hlfir.region_assign {
    %4 = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
    %5 = fir.call @_QPnew_obja() fastmath<contract> : () -> !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    fir.save_result %5 to %0(%2) : !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>
    %6:2 = hlfir.declare %0(%2) {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
    hlfir.yield %6#0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>> cleanup {
      %7 = fir.embox %0(%2) : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
      %8 = fir.convert %7 : (!fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.box<none>
      %9 = fir.call @_FortranADestroy(%8) fastmath<contract> : (!fir.box<none>) -> none
      fir.call @llvm.stackrestore.p0(%4) fastmath<contract> : (!fir.ref<i8>) -> ()
    }
  } to {
    hlfir.yield %3#0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
  } user_defined_assign  (%arg0: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) to (%arg1: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) {
    %4 = fir.embox %arg1 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %5 = fir.convert %4 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %6 = fir.embox %arg0 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    %7 = fir.convert %6 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
    fir.call @_QMtypesPassign(%5, %7) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @_QPtest2() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>> {bindc_name = ".result"}
// CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFtest2Exa) : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_3]]) {uniq_name = "_QFtest2Exa"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
// CHECK:           %[[VAL_5:.*]] = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
// CHECK:           %[[VAL_6:.*]] = fir.call @_QPnew_obja() fastmath<contract> : () -> !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           fir.save_result %[[VAL_6]] to %[[VAL_1]](%[[VAL_3]]) : !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
// CHECK:           %[[VAL_8:.*]] = hlfir.as_expr %[[VAL_7]]#0 : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !hlfir.expr<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]](%[[VAL_3]]) {uniq_name = ".tmp.assign"} : (!hlfir.expr<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, i1)
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_10]] to %[[VAL_0]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             %[[VAL_14:.*]] = fir.embox %[[VAL_12]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             %[[VAL_16:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:             fir.call @_QMtypesPassign(%[[VAL_15]], %[[VAL_17]]) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
// CHECK:           }
// CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, i1
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_20:.*]] = fir.call @_FortranADestroy(%[[VAL_19]]) fastmath<contract> : (!fir.box<none>) -> none
// CHECK:           fir.call @llvm.stackrestore.p0(%[[VAL_5]]) fastmath<contract> : (!fir.ref<i8>) -> ()
// CHECK:           return
// CHECK:         }

//! subroutine test3(y)
//!   use types
//!   interface
//!      function new_obja()
//!        use types
//!        type(ud_assign) :: new_obja(2)
//!      end function new_obja
//!   end interface
//!   integer :: y(2)
//!   type(ud_assign), save :: x(2)
//!   where (y == 1) x = new_obja()
//! end subroutine test3
func.func @_QPtest3(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "y"}) {
  %c1_i32 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %0 = fir.alloca !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>> {bindc_name = ".result"}
  %1 = fir.address_of(@_QFtest3Ex) : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
  %2 = fir.shape %c2 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
  %4:2 = hlfir.declare %arg0(%2) {uniq_name = "_QFtest3Ey"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  hlfir.where {
    %5 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<2x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %6 = hlfir.designate %4#0 (%arg1)  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
      %7 = fir.load %6 : !fir.ref<i32>
      %8 = arith.cmpi eq, %7, %c1_i32 : i32
      %9 = fir.convert %8 : (i1) -> !fir.logical<4>
      hlfir.yield_element %9 : !fir.logical<4>
    }
    hlfir.yield %5 : !hlfir.expr<2x!fir.logical<4>> cleanup {
      hlfir.destroy %5 : !hlfir.expr<2x!fir.logical<4>>
    }
  } do {
    hlfir.region_assign {
      %5 = hlfir.exactly_once : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>> {
        %7 = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
        %8 = fir.call @_QPnew_obja() fastmath<contract> : () -> !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
        fir.save_result %8 to %0(%2) : !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>
        %9:2 = hlfir.declare %0(%2) {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
        hlfir.yield %9#0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>> cleanup {
          %10 = fir.embox %0(%2) : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
          %11 = fir.convert %10 : (!fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.box<none>
          %12 = fir.call @_FortranADestroy(%11) fastmath<contract> : (!fir.box<none>) -> none
          fir.call @llvm.stackrestore.p0(%7) fastmath<contract> : (!fir.ref<i8>) -> ()
        }
      }
      %6:2 = hlfir.declare %5(%2) {uniq_name = ".func.pointer.result"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
      hlfir.yield %6#0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
    } to {
      hlfir.yield %3#0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
    } user_defined_assign  (%arg1: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) to (%arg2: !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) {
      %5 = fir.embox %arg2 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
      %6 = fir.convert %5 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
      %7 = fir.embox %arg1 : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
      %8 = fir.convert %7 : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
      fir.call @_QMtypesPassign(%6, %8) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
    }
  }
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
// CHECK:           %[[VAL_2:.*]] = fir.alloca i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>> {bindc_name = ".result"}
// CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QFtest3Ex) : !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_7]]) {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) {uniq_name = "_QFtest3Ey"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
// CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<2x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:             %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_13]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_15]] : !fir.logical<4>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_10]](%[[VAL_7]]) {uniq_name = ".tmp.where"} : (!hlfir.expr<2x!fir.logical<4>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.ref<!fir.array<2x!fir.logical<4>>>, i1)
// CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<2x!fir.logical<4>>
// CHECK:           %[[VAL_18:.*]] = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
// CHECK:           %[[VAL_19:.*]] = fir.call @_QPnew_obja() fastmath<contract> : () -> !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:           fir.save_result %[[VAL_19]] to %[[VAL_5]](%[[VAL_7]]) : !fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_7]]) {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>)
// CHECK:           %[[VAL_27:.*]] = fir.call @_FortranACreateValueStack(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK:           fir.do_loop %[[VAL_28:.*]] = %{{.*}} to %[[VAL_4]] step %{{.*}} {
// CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_28]])  : (!fir.ref<!fir.array<2x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_31]] {
// CHECK:               %[[VAL_20B:.*]]:2 = hlfir.declare %[[VAL_20]]#0(%[[VAL_7]]) {uniq_name = ".func.pointer.result"}
// CHECK:               %[[VAL_32:.*]] = hlfir.designate %[[VAL_20B]]#0 (%[[VAL_28]])  : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_33:.*]] = fir.embox %[[VAL_32]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<none>
// CHECK:               %[[VAL_35:.*]] = fir.call @_FortranAPushValue(%[[VAL_27]], %[[VAL_34]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
// CHECK:             }
// CHECK:           }
// CHECK:           fir.do_loop %[[VAL_37:.*]] = %{{.*}} to %[[VAL_4]] step %{{.*}} {
// CHECK:             %[[VAL_38:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_37]])  : (!fir.ref<!fir.array<2x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_40]] {
// CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %{{.*}} : i64
// CHECK:               fir.store %[[VAL_42]] to %[[VAL_2]] : !fir.ref<i64>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_44:.*]] = fir.call @_FortranAValueAt(%[[VAL_27]], %[[VAL_41]], %[[VAL_43]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> none
// CHECK:               %[[VAL_45:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>>
// CHECK:               %[[VAL_46:.*]] = fir.box_addr %[[VAL_45]] : (!fir.box<!fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_47:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_37]])  : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_49:.*]] = fir.embox %[[VAL_47]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_51:.*]] = fir.embox %[[VAL_48]] : (!fir.ref<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (!fir.box<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>
// CHECK:               fir.call @_QMtypesPassign(%[[VAL_50]], %[[VAL_52]]) fastmath<contract> : (!fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>, !fir.class<!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<!fir.array<2x!fir.logical<4>>>, i1
// CHECK:           %[[VAL_53:.*]] = fir.call @_FortranADestroyValueStack(%[[VAL_27]]) : (!fir.llvm_ptr<i8>) -> none
// CHECK:           %[[VAL_54:.*]] = fir.embox %[[VAL_5]](%[[VAL_7]]) : (!fir.ref<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>
// CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (!fir.box<!fir.array<2x!fir.type<_QMtypesTud_assign{x:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_56:.*]] = fir.call @_FortranADestroy(%[[VAL_55]]) fastmath<contract> : (!fir.box<none>) -> none
// CHECK:           fir.call @llvm.stackrestore.p0(%[[VAL_18]]) fastmath<contract> : (!fir.ref<i8>) -> ()
// CHECK:           return
// CHECK:         }
