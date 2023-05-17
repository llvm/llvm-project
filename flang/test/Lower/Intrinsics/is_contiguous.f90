! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_is_contiguous(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4> {adapt.valuebyref}
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {adapt.valuebyref}
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p", uniq_name = "_QFtest_is_contiguousEp"}
! CHECK:         %[[VAL_42:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_43:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_42]]) {{.*}} : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_44]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:         fir.call @_QPfoo1(%[[VAL_2]]) {{.*}} : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:         %[[VAL_45:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:         %[[VAL_47:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_46]]) {{.*}} : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_48]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:         fir.call @_QPfoo2(%[[VAL_1]]) {{.*}} : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine test_is_contiguous(a)
  real :: a(:)
  real, pointer :: p(:)

  call bar(p)

  call foo1(is_contiguous(a))
  call foo2(is_contiguous(p))
end
