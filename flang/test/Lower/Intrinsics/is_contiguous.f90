! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_is_contiguous(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[ADECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_is_contiguousEa"}
! CHECK:         %[[PDECL:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_is_contiguousEp"}
! CHECK:         %[[VAL_42:.*]] = fir.convert %[[ADECL]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_43:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_42]]) {{.*}} : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i1) -> !fir.logical<4>
! CHECK:         %[[ASSOC1:.*]]:3 = hlfir.associate %[[VAL_44]] {adapt.valuebyref}
! CHECK:         fir.call @_QPfoo1(%[[ASSOC1]]#0) {{.*}} : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:         hlfir.end_associate %[[ASSOC1]]#1, %[[ASSOC1]]#2
! CHECK:         %[[VAL_45:.*]] = fir.load %[[PDECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:         %[[VAL_47:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_46]]) {{.*}} : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i1) -> !fir.logical<4>
! CHECK:         %[[ASSOC2:.*]]:3 = hlfir.associate %[[VAL_48]] {adapt.valuebyref}
! CHECK:         fir.call @_QPfoo2(%[[ASSOC2]]#0) {{.*}} : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:         hlfir.end_associate %[[ASSOC2]]#1, %[[ASSOC2]]#2
! CHECK:         return
! CHECK:       }

subroutine test_is_contiguous(a)
  real :: a(:)
  real, pointer :: p(:)

  call bar(p)

  call foo1(is_contiguous(a))
  call foo2(is_contiguous(p))
end
