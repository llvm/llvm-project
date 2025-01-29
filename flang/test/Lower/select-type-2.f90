! RUN: bbc -emit-fir -hlfir=false %s -o - | fir-opt --fir-polymorphic-op | FileCheck %s
module select_type_2
  type p1
    integer :: a
    integer :: b
  end type

  type, extends(p1) :: p2
    integer :: c
  end type

  type, extends(p2) :: p3
    integer :: d
  end type

contains

  subroutine select_type1(a)
    class(p1), intent(in) :: a

    select type (a)
    class is (p1)
      print*, 'class is p1'
    class is (p3)
      print*, 'class is p3'
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_2Pselect_type1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_2Tp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CHECK:      %[[TDESC_P3_ADDR:.*]] = fir.address_of(@_QMselect_type_2E.dt.p3) : !fir.ref<!fir.type<{{.*}}>>
! CHECK:      %[[TDESC_P3_CONV:.*]] = fir.convert %[[TDESC_P3_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_2Tp1{a:i32,b:i32}>>) -> !fir.box<none>
! CHECK:      %[[CLASS_IS_CMP:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P3_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CHECK:      cf.cond_br %[[CLASS_IS_CMP]], ^[[CLASS_IS_P3_BLK:.*]], ^[[NOT_CLASS_IS_P3_BLK:.*]]
! CHECK:    ^bb[[NOT_CLASS_IS_P1:[0-9]]]:
! CHECK:      cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CHECK:    ^bb[[CLASS_IS_P1:[0-9]]]:
! CHECK:      cf.br ^bb[[END_SELECT_BLK:[0-9]]]
! CHECK:    ^[[NOT_CLASS_IS_P3_BLK]]:
! CHECK:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_2E.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_2Tp1{a:i32,b:i32}>>) -> !fir.box<none>
! CHECK:      %[[CLASS_IS_CMP:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CHECK:      cf.cond_br %[[CLASS_IS_CMP]], ^bb[[CLASS_IS_P1]], ^bb[[NOT_CLASS_IS_P1]]
! CHECK:    ^[[CLASS_IS_P3_BLK]]:
! CHECK:      cf.br ^bb[[END_SELECT_BLK]]
! CHECK:    ^bb[[DEFAULT_BLK]]:
! CHECK:      cf.br ^bb[[END_SELECT_BLK]]
! CHECK:    ^bb[[END_SELECT_BLK]]:
! CHECK:      return

  end module
