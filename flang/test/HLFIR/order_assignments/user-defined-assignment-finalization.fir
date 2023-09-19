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
