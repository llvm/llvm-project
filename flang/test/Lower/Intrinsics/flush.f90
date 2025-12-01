! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
!
! Test lowering of intrinsic subroutine FLUSH with and without optional UNIT argument.
!
! CHECK-LABEL: func.func @_QPflush_all()
! CHECK: %[[UNIT:.*]] = arith.constant -1 : i32
! CHECK: fir.call @_FortranAFlush(%[[UNIT]]) fastmath<contract> : (i32) -> ()
! CHECK: return
subroutine flush_all()
  call flush() ! flush all units
end subroutine

! CHECK-LABEL: func.func @_QPflush_unit()
! CHECK: %[[ALLOCA:.*]] = fir.alloca i32
! CHECK: %[[UNITC:.*]] = arith.constant 10 : i32
! CHECK: fir.store %[[UNITC]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK: %[[LOADED:.*]] = fir.load %[[ALLOCA]] : !fir.ref<i32>
! CHECK: fir.call @_FortranAFlush(%[[LOADED]]) fastmath<contract> : (i32) -> ()
! CHECK: return
subroutine flush_unit()
  call flush(10) ! flush specific unit
end subroutine

! CHECK-LABEL: func.func @_QPflush_optional(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "unit", fir.optional}) {
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFflush_optionalEunit"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.ref<i32>) -> i1
! CHECK: %[[UNIT:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
! CHECK:   %[[LOADED:.*]] = fir.load %[[DECL]]#0 : !fir.ref<i32>
! CHECK:   fir.result %[[LOADED]] : i32
! CHECK: } else {
! CHECK:   %[[DEFAULT:.*]] = arith.constant -1 : i32
! CHECK:   fir.result %[[DEFAULT]] : i32
! CHECK: }
! CHECK: fir.call @_FortranAFlush(%[[UNIT]]) fastmath<contract> : (i32) -> ()
! CHECK: return
subroutine flush_optional(unit)
  integer, optional :: unit
  call flush(unit) ! flush with dynamically optional argument
end subroutine
