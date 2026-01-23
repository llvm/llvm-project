! Test lowering of COPYPRIVATE with character arguments
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Testcase from: https://github.com/llvm/llvm-project/issues/142123

! CHECK-LABEL:  func.func private @_copy_boxchar_c8xU(
! CHECK-SAME:     %arg0: [[TYPE:!fir.ref<!fir.boxchar<1>>]],
! CHECK-SAME:     %arg1: [[TYPE]]) attributes {llvm.linkage = #llvm.linkage<internal>} {
! CHECK:    %[[RDST:.*]] = fir.load %arg0 : [[TYPE]]
! CHECK:    %[[RSRC:.*]] = fir.load %arg1 : [[TYPE]]
! CHECK:    %[[UDST:.*]]:2 = fir.unboxchar %[[RDST:.*]] : ([[UTYPE:!fir.boxchar<1>]]) -> ([[RTYPE:!fir.ref<!fir.char<1,\?>>]], [[ITYPE:index]])
! CHECK:    %[[USRC:.*]]:2 = fir.unboxchar %[[RSRC:.*]] : ([[UTYPE]]) -> ([[RTYPE]], [[ITYPE]])
! CHECK:    %[[DST:.*]]:2 = hlfir.declare %[[UDST:.*]]#0 typeparams %[[UDST:.*]]#1 {uniq_name = "[[NAME1:.*]]"} : ([[RTYPE]], [[ITYPE]]) -> ([[UTYPE]], [[RTYPE]])
! CHECK:    %[[SRC:.*]]:2 = hlfir.declare %[[USRC:.*]]#0 typeparams %[[UDST:.*]]#1 {uniq_name = "[[NAME2:.*]]"} : ([[RTYPE]], [[ITYPE]]) -> ([[UTYPE]], [[RTYPE]])
! CHECK:    hlfir.assign %[[SRC:.*]]#0 to %[[DST:.*]]#0 : [[UTYPE]], [[UTYPE]]
! CHECK:    return
! CHECK:  }

! CHECK-LABEL: func.func @_QPs(%arg0: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK: %[[ALLOC:.*]] = fir.alloca !fir.boxchar<1>
! CHECK: fir.store %[[SRC:.*]] to %[[ALLOC:.*]] : !fir.ref<!fir.boxchar<1>>
! CHECK: omp.single copyprivate([[ALLOC:.*]] -> @_copy_boxchar_c8xU : !fir.ref<!fir.boxchar<1>>) {
! CHECK:   hlfir.assign %[[NEW_VAL:.*]] to %[[SRC:.*]] : !fir.ref<!fir.char<1,3>>, !fir.boxchar<1>
! CHECK:   omp.terminator
! CHECK: }

subroutine s(c)
character(*) :: c
!$omp single copyprivate(c)
c = "bar"
!$omp end single
end subroutine

character(len=3) :: c
call s(c)
end
