! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | fir-opt --fir-polymorphic-op | FileCheck --check-prefix=CFG %s
module select_type_lower_test
  type p1
    integer :: a
    integer :: b
  end type

  type, extends(p1) :: p2
    integer :: c
  end type

  type, extends(p1) :: p3(k)
    integer, kind :: k
    real(k) :: r
  end type

  type, extends(p2) :: p4
    integer :: d
  end type

  type :: p5
    integer :: a
  contains
    procedure :: negate
    generic :: operator(-) => negate
  end type

contains

  function get_class()
    class(p1), pointer :: get_class
  end function

  function negate(this)
    class(p5), intent(in) :: this
    class(p5), allocatable :: negate
    allocate(negate, source=this)
    negate%a = -this%a
  end function

  subroutine select_type1(a)
    class(p1), intent(in) :: a

    select type (a)
    type is (p1)
      print*, 'type is p1'
    class is (p1)
      print*, 'class is p1'
    class is (p2)
      print*, 'class is p2', a%c
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"})
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[A]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_P1_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{{.*}}>>, ^[[CLASS_IS_P2_BLK:.*]], unit, ^[[DEFAULT_BLOCK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_P1_BLK]]
! CHECK: ^[[CLASS_IS_P2_BLK]]
! CHECK: %[[P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp2{{.*}}>>
! CHECK: %[[P2DECL:.*]]:2 = hlfir.declare %[[P2]]
! CHECK: %{{.*}} = hlfir.designate %[[P2DECL]]#0{"c"} {{.*}}-> !fir.ref<i32>
! CHECK: ^[[DEFAULT_BLOCK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type1(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CFG: %[[TDESC_P1_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[BOX_TDESC:.*]] = fir.box_tdesc %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG: %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> index
! CFG: %[[BOX_TDESC_CONV:.*]] = fir.convert %[[BOX_TDESC]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> index
! CFG: %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG: cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG: ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG: %[[TDESC_P2_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{{.*}}
! CFG: %[[TDESC_P2_CONV:.*]] = fir.convert %[[TDESC_P2_ADDR]] : (!fir.tdesc<{{.*}}>) -> !fir.ref<none>
! CFG: %[[BOX_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG: %[[CLASS_IS_P2:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P2_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG: cf.cond_br %[[CLASS_IS_P2]], ^bb[[CLASS_IS_P2_BLK:.*]], ^[[NOT_CLASS_IS_P2_BLK:.*]]
! CFG: ^[[TYPE_IS_P1_BLK]]:
! CFG: cf.br ^bb[[EXIT_SELECT_BLK:[0-9]]]
! CFG: ^bb[[NOT_CLASS_IS_P1_BLK:[0-9]]]:
! CFG: cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CFG: ^bb[[CLASS_IS_P1_BLK:[0-9]]]:
! CFG: cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG: ^[[NOT_CLASS_IS_P2_BLK]]:
! CFG: %[[TDESC_P1_ADDR2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[TDESC_P1_CONV2:.*]] = fir.convert %[[TDESC_P1_ADDR2]] : (!fir.tdesc<{{.*}}>) -> !fir.ref<none>
! CFG: %[[BOX_NONE2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG: %[[CLASS_IS_P1:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE2]], %[[TDESC_P1_CONV2]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG: cf.cond_br %[[CLASS_IS_P1]], ^bb[[CLASS_IS_P1_BLK]], ^bb[[NOT_CLASS_IS_P1_BLK]]
! CFG: ^bb[[CLASS_IS_P2_BLK]]:
! CFG: cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG: ^bb[[DEFAULT_BLK]]:
! CFG: cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG: ^bb[[EXIT_SELECT_BLK]]:
! CFG: return

  subroutine select_type2()
    select type (a => get_class())
    type is (p1)
      print*, 'type is p1'
    class is (p1)
      print*, 'class is p1'
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type2()
! CHECK: %[[RESULT:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {bindc_name = ".result"}
! CHECK: %[[FCTCALL:.*]] = fir.call @_QMselect_type_lower_testPget_class() {{.*}}: () -> !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: fir.save_result %[[FCTCALL]] to %[[RESULT]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[RESULT]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>)
! CHECK: %[[SELECTOR:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_BLK]]
! CHECK: ^[[DEFAULT_BLK]]
! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type2() {
! CFG: %[[CLASS_ALLOCA:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {bindc_name = ".result"}
! CFG: %[[GET_CLASS:.*]] = fir.call @_QMselect_type_lower_testPget_class() {{.*}} : () -> !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG: fir.save_result %[[GET_CLASS]] to %[[CLASS_ALLOCA]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG: %[[DECL:.*]]:2 = hlfir.declare %[[CLASS_ALLOCA]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>)
! CFG: %[[LOAD_CLASS:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG: %[[TDESC_P1_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[CLASS_TDESC:.*]] = fir.box_tdesc %[[LOAD_CLASS]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<{{.*}}>
! CFG: %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.tdesc<{{.*}}>) -> index
! CFG: %[[BOX_TDESC_CONV:.*]] = fir.convert %[[CLASS_TDESC]] : (!fir.tdesc<{{.*}}>) -> index
! CFG: %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG: cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG: ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG: %[[TDESC_P1_ADDR2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[TDESC_P1_REF:.*]] = fir.convert %[[TDESC_P1_ADDR2]] : (!fir.tdesc<{{.*}}>) -> !fir.ref<none>
! CFG: %[[BOX_NONE:.*]] = fir.convert %[[LOAD_CLASS]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CFG: %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_REF]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG: cf.cond_br %[[CLASS_IS]], ^[[CLASS_IS_BLK:.*]], ^[[NOT_CLASS_IS_BLK:.*]]
! CFG: ^[[TYPE_IS_P1_BLK]]:
! CFG: cf.br ^[[END_SELECT_BLK:.*]]
! CFG: ^[[NOT_CLASS_IS_BLK]]:
! CFG: cf.br ^[[DEFAULT_BLK:.*]]
! CFG: ^[[CLASS_IS_BLK]]:
! CFG: cf.br ^[[END_SELECT_BLK]]
! CFG: ^[[DEFAULT_BLK]]:
! CFG: cf.br ^[[END_SELECT_BLK]]
! CFG: ^[[END_SELECT_BLK]]:
! CFG: return

  subroutine select_type3(a)
    class(p1), pointer, intent(in) :: a(:)

    select type (x => a(1))
    type is (p1)
      print*, 'type is p1'
    class is (p1)
      print*, 'class is p1'
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"})
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_in, pointer>, uniq_name = "_QMselect_type_lower_testFselect_type3Ea"}
! CHECK: %[[ARG0_LOAD:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[SELECTOR:.*]] = hlfir.designate %[[ARG0_LOAD]] (%{{.*}})  : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_BLK]]
! CHECK: ^[[DEFAULT_BLK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"}) {
! CFG:      %[[SELECTOR:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:      %[[SELECTOR_TDESC:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<{{.*}}>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.tdesc{{.*}}>) -> index
! CFG:      %[[TDESC_CONV:.*]] = fir.convert %[[SELECTOR_TDESC]] : (!fir.tdesc<{{.*}}>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG:    ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG:      %[[TDESC_P1_ADDR2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:      %[[TDESC_P1_CONV2:.*]] = fir.convert %[[TDESC_P1_ADDR2]] : (!fir.tdesc{{.*}}>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG:      %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV2]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS]], ^[[CLASS_IS_BLK:.*]], ^[[NOT_CLASS_IS:.*]]
! CFG:    ^[[TYPE_IS_P1_BLK]]:
! CFG:      cf.br ^bb[[END_SELECT_BLK:[0-9]]]
! CFG:    ^[[NOT_CLASS_IS]]:
! CFG:      cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CFG:    ^[[CLASS_IS_BLK]]:
! CFG:      cf.br ^bb[[END_SELECT_BLK]]
! CFG:    ^bb[[DEFAULT_BLK]]:
! CFG:      cf.br ^bb[[END_SELECT_BLK]]
! CFG:    ^bb[[END_SELECT_BLK]]:
! CFG:      return

  subroutine select_type4(a)
    class(p1), intent(in) :: a
    select type(a)
    type is(p3(8))
      print*, 'type is p3(8)'
    type is(p3(4))
      print*, 'type is p3(4)'
    class is (p1)
      print*, 'class is p1'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"})
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope {{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMselect_type_lower_testFselect_type4Ea"}
! CHECK: fir.select_type %[[A]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp3K8{{.*}}>>, ^[[P3_8:.*]], #fir.type_is<!fir.type<_QMselect_type_lower_testTp3K4{{.*}}>>, ^[[P3_4:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[P1:.*]], unit, ^[[EXIT:.*]]]
! CHECK: ^[[P3_8]]
! CHECK: ^[[P3_4]]
! CHECK: ^[[P1]]
! CHECK: ^[[EXIT]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CFG:      %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope {{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMselect_type_lower_testFselect_type4Ea"}
! CFG:      %[[TDESC_P3_8_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp3K8
! CFG:      %[[BOX_TDESC:.*]] = fir.box_tdesc %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<{{.*}}>
! CFG:      %[[TDESC_P3_8_CONV:.*]] = fir.convert %[[TDESC_P3_8_ADDR]] : (!fir.tdesc{{.*}}>) -> index
! CFG:      %[[BOX_TDESC_CONV:.*]] = fir.convert %[[BOX_TDESC]] : (!fir.tdesc<{{.*}}>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P3_8_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[P3_8_BLK:.*]], ^[[NOT_P3_8_BLK:.*]]
! CFG:    ^[[NOT_P3_8_BLK]]:
! CFG:      %[[TDESC_P3_4_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp3K4
! CFG:      %[[BOX_TDESC_2:.*]] = fir.box_tdesc %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<{{.*}}>
! CFG:      %[[TDESC_P3_4_CONV:.*]] = fir.convert %[[TDESC_P3_4_ADDR]] : (!fir.tdesc{{.*}}>) -> index
! CFG:      %[[BOX_TDESC_2_CONV:.*]] = fir.convert %[[BOX_TDESC_2]] : (!fir.tdesc<{{.*}}>) -> index
! CFG:      %[[TDESC_CMP_2:.*]] = arith.cmpi eq, %[[TDESC_P3_4_CONV]], %[[BOX_TDESC_2_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP_2]], ^[[P3_4_BLK:.*]], ^[[NOT_P3_4_BLK:.*]]
! CFG:    ^[[P3_8_BLK]]:
! CFG:      _FortranAioOutputAscii
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK:[0-9]]]
! CFG:    ^[[NOT_P3_4_BLK]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.tdesc{{.*}}>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG:      %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS]], ^[[P1_BLK:.*]], ^[[NOT_P1_BLK:.*]]
! CFG:    ^[[P3_4_BLK]]:
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG:    ^[[NOT_P1_BLK]]:
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG:    ^[[P1_BLK]]:
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK]]
! CFG:    ^bb[[EXIT_SELECT_BLK]]:
! CFG:      return

  subroutine select_type5(a)
    class(*), intent(in) :: a

    select type (x => a)
    type is (integer(1))
      print*, 'type is integer(1)'
    type is (integer(4))
      print*, 'type is integer(4)'
    type is (real(4))
      print*, 'type is real'
    type is (logical)
      print*, 'type is logical'
    type is (character(*))
      print*, 'type is character'
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMselect_type_lower_testFselect_type5Ea"} : (!fir.class<none>, !fir.dscope) -> (!fir.class<none>, !fir.class<none>)
! CHECK: fir.select_type %[[A]]#1 : !fir.class<none>
! CHECK-SAME: [#fir.type_is<i8>, ^[[I8_BLK:.*]], #fir.type_is<i32>, ^[[I32_BLK:.*]], #fir.type_is<f32>, ^[[F32_BLK:.*]], #fir.type_is<!fir.logical<4>>, ^[[LOG_BLK:.*]], #fir.type_is<!fir.char<1,?>>, ^[[CHAR_BLK:.*]], unit, ^[[DEFAULT:.*]]]
! CHECK: ^[[I8_BLK]]
! CHECK: ^[[I32_BLK]]
! CHECK: ^[[F32_BLK]]
! CHECK: ^[[LOG_BLK]]
! CHECK: ^[[CHAR_BLK]]
! CHECK: ^[[DEFAULT]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}) {
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMselect_type_lower_testFselect_type5Ea"} : (!fir.class<none>, !fir.dscope) -> (!fir.class<none>, !fir.class<none>)
! CFG: %[[INT8_TC:.*]] = arith.constant 7 : i8
! CFG: %[[TYPE_CODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG: %[[IS_INT8:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[INT8_TC]] : i8
! CFG: cf.cond_br %[[IS_INT8]], ^[[INT8_BLK:.*]], ^[[NOT_INT8:.*]]
! CFG: ^[[NOT_INT8]]:
! CFG: %[[INT32_TC:.*]] = arith.constant 9 : i8
! CFG: %[[TYPE_CODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG: %[[IS_INT32:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[INT32_TC]] : i8
! CFG: cf.cond_br %[[IS_INT32]], ^[[INT32_BLK:.*]], ^[[NOT_INT32_BLK:.*]]
! CFG: ^[[INT8_BLK]]:
! CFG: cf.br ^[[EXIT_BLK:.*]]
! CFG: ^[[NOT_INT32_BLK]]:
! CFG: %[[FLOAT_TC:.*]] = arith.constant 27 : i8
! CFG: %[[TYPE_CODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG: %[[IS_FLOAT:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[FLOAT_TC]] : i8
! CFG: cf.cond_br %[[IS_FLOAT]], ^[[FLOAT_BLK:.*]], ^[[NOT_FLOAT_BLK:.*]]
! CFG: ^[[INT32_BLK]]:
! CFG: cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_FLOAT_BLK]]:
! CFG: %[[LOGICAL_TC:.*]] = arith.constant 14 : i8
! CFG: %[[TYPE_CODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG: %[[IS_LOGICAL:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[LOGICAL_TC]] : i8
! CFG: cf.cond_br %[[IS_LOGICAL]], ^[[LOGICAL_BLK:.*]], ^[[NOT_LOGICAL_BLK:.*]]
! CFG: ^[[FLOAT_BLK]]:
! CFG: cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_LOGICAL_BLK]]:
! CFG: %[[CHAR_TC:.*]] = arith.constant 40 : i8
! CFG: %[[TYPE_CODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG: %[[IS_CHAR:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[CHAR_TC]] : i8
! CFG: cf.cond_br %[[IS_CHAR]], ^[[CHAR_BLK:.*]], ^[[NOT_CHAR_BLK:.*]]
! CFG: ^[[LOGICAL_BLK]]:
! CFG: cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_CHAR_BLK]]:
! CFG: cf.br ^[[DEFAULT_BLK:.*]]
! CFG: ^[[CHAR_BLK]]:
! CFG: cf.br ^[[EXIT_BLK]]
! CFG: ^[[DEFAULT_BLK]]:
! CFG: cf.br ^[[EXIT_BLK]]
! CFG: ^[[EXIT_BLK]]:
! CFG: return

  subroutine select_type6(a)
    class(*) :: a

    select type(a)
    type is (integer)
      a = 100
    type is (real)
      a = 2.0
    class default
      stop 'error'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[A]]#1 : !fir.class<none> [#fir.type_is<i32>, ^[[INT_BLK:.*]], #fir.type_is<f32>, ^[[REAL_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[INT_BLK]]
! CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<i32>
! CHECK:  %[[DECL_INT:.*]]:2 = hlfir.declare %[[BOX_ADDR]]
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  hlfir.assign %[[C100]] to %[[DECL_INT]]#0 : i32, !fir.ref<i32>

! CHECK: ^[[REAL_BLK]]:  // pred: ^bb0
! CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<f32>
! CHECK:  %[[DECL_REAL:.*]]:2 = hlfir.declare %[[BOX_ADDR]]
! CHECK:  %[[C2:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:  hlfir.assign %[[C2]] to %[[DECL_REAL]]#0 : f32, !fir.ref<f32>


! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CFG:   %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CFG:   %[[INT32_TYPECODE:.*]] = arith.constant 9 : i8
! CFG:   %[[ARG0_TYPECODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG:   %[[IS_TYPECODE:.*]] = arith.cmpi eq, %[[ARG0_TYPECODE]], %[[INT32_TYPECODE]] : i8
! CFG:   cf.cond_br %[[IS_TYPECODE]], ^[[TYPE_IS_INT_BLK:.*]], ^[[TYPE_NOT_INT_BLK:.*]]
! CFG: ^[[TYPE_NOT_INT_BLK]]:
! CFG:   %[[FLOAT_TYPECODE:.*]] = arith.constant 27 : i8
! CFG:   %[[ARG0_TYPECODE:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<none>) -> i8
! CFG:   %[[IS_TYPECODE:.*]] = arith.cmpi eq, %[[ARG0_TYPECODE]], %[[FLOAT_TYPECODE]] : i8
! CFG:   cf.cond_br %[[IS_TYPECODE]], ^[[TYPE_IS_REAL_BLK:.*]], ^[[TYPE_NOT_REAL_BLK:.*]]
! CFG: ^[[TYPE_IS_INT_BLK]]:
! CFG:   %[[BOX_ADDR:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<i32>
! CFG:   %[[DECL_INT:.*]]:2 = hlfir.declare %[[BOX_ADDR]]
! CFG:   %[[C100:.*]] = arith.constant 100 : i32
! CFG:   hlfir.assign %[[C100]] to %[[DECL_INT]]#0 : i32, !fir.ref<i32>
! CFG:   cf.br ^[[EXIT_SELECT_BLK:.*]]
! CFG: ^[[TYPE_NOT_REAL_BLK]]:
! CFG:   cf.br ^[[DEFAULT_BLK:.*]]
! CFG: ^[[TYPE_IS_REAL_BLK]]:
! CFG:   %[[BOX_ADDR:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<f32>
! CFG:   %[[DECL_REAL:.*]]:2 = hlfir.declare %[[BOX_ADDR]]
! CFG:   %[[CST:.*]] = arith.constant 2.000000e+00 : f32
! CFG:   hlfir.assign %[[CST]] to %[[DECL_REAL]]#0 : f32, !fir.ref<f32>
! CFG:   cf.br ^[[EXIT_SELECT_BLK]]
! CFG: ^[[DEFAULT_BLK]]:
! CFG:   fir.call @_FortranAStopStatementText
! CFG:   fir.unreachable
! CFG: ^[[EXIT_SELECT_BLK]]:
! CFG:   return

  subroutine select_type7(a)
    class(*), intent(out) :: a

    select type(a)
    class is (p1)
      print*, 'CLASS IS P1'
    class is (p2)
      print*, 'CLASS IS P2'
    class is (p4)
      print*, 'CLASS IS P4'
    class default
      print*, 'CLASS DEFAULT'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DSCOPE]] arg 1 {{.*}}uniq_name = "_QMselect_type_lower_testFselect_type7Ea"{{.*}} : (!fir.class<none>, !fir.dscope) -> (!fir.class<none>, !fir.class<none>)
! CHECK: fir.select_type %[[DECL]]#1 :
! CHECK-SAME: !fir.class<none> [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{{.*}}>>, ^bb2, #fir.class_is<!fir.type<_QMselect_type_lower_testTp4{{.*}}>>, ^bb3, unit, ^bb4]

! Check correct ordering of class is type guard. The expected flow should be:
!   class is (p4) -> class is (p2) -> class is (p1) -> class default

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}) {
! CFG:      %[[TDESC_P4_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp4
! CFG:      %[[TDESC_P4_CONV:.*]] = fir.convert %[[TDESC_P4_ADDR]] : (!fir.tdesc{{.*}}>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P4:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P4_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P4]], ^[[CLASS_IS_P4_BLK:.*]], ^[[CLASS_NOT_P4_BLK:.*]]
! CFG:    ^bb[[CLASS_NOT_P1_BLK:[0-9]]]:
! CFG:      cf.br ^[[CLASS_DEFAULT_BLK:.*]]
! CFG:    ^bb[[CLASS_IS_P1_BLK:[0-9]]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK:.*]]
! CFG:    ^bb[[CLASS_NOT_P2_BLK:[0-9]]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.tdesc{{.*}}>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P1:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P1]], ^bb[[CLASS_IS_P1_BLK]], ^bb[[CLASS_NOT_P1_BLK]]
! CFG:    ^bb[[CLASS_IS_P2_BLK:[0-9]]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:    ^[[CLASS_NOT_P4_BLK]]:
! CFG:      %[[TDESC_P2_ADDR:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2
! CFG:      %[[TDESC_P2_CONV:.*]] = fir.convert %[[TDESC_P2_ADDR]] : (!fir.tdesc{{.*}}>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %{{.*}} : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P2:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P2_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P2]], ^bb[[CLASS_IS_P2_BLK]], ^bb[[CLASS_NOT_P2_BLK]]
! CFG:   ^[[CLASS_IS_P4_BLK]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:   ^[[CLASS_DEFAULT_BLK]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:   ^[[EXIT_SELECT_BLK]]:
! CFG:      return

  subroutine select_type8(a)
    class(*) :: a(:)

    select type(a)
    type is (integer)
      a = 100
    type is (real)
      a = 2.0
    type is (character(*))
      a(1) = 'c'
      a(2) = 'h'
    type is (p1)
      a%a = 1
      a%b = 2
    class is(p2)
      a%a = 1
      a%b = 2
      a%c = 3
    class default
      stop 'error'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type8(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "a"}) {
! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]]{{.*}} {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.class<!fir.array<?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CHECK: fir.select_type %[[A]]#1 : !fir.class<!fir.array<?xnone>> [#fir.type_is<i32>, ^bb{{.*}}, #fir.type_is<f32>, ^bb{{.*}}, #fir.type_is<!fir.char<1,?>>, ^bb{{.*}}, #fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb{{.*}}, unit, ^bb{{.*}}]
! CHECK: ^bb{{.*}}:
! CHECK:   %[[BOX_I32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[DECL_I32:.*]]:2 = hlfir.declare %[[BOX_I32]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:   %[[C100:.*]] = arith.constant 100 : i32
! CHECK:   hlfir.assign %[[C100]] to %[[DECL_I32]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:   %[[BOX_F32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:   %[[DECL_F32:.*]]:2 = hlfir.declare %[[BOX_F32]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:   %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:   hlfir.assign %[[CST2]] to %[[DECL_F32]]#0 : f32, !fir.box<!fir.array<?xf32>>
! CHECK:   cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:   %[[BOX_CHR:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:   %[[DECL_CHR:.*]]:2 = hlfir.declare %[[BOX_CHR]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:   %[[CHR_C:.*]] = fir.address_of(@_QQclX63) : !fir.ref<!fir.char<1>>
! CHECK:   %[[DESIG1:.*]] = hlfir.designate %[[DECL_CHR]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:   hlfir.assign %{{.*}} to %[[DESIG1]] : !fir.ref<!fir.char<1>>, !fir.boxchar<1>
! CHECK:   %[[CHR_H:.*]] = fir.address_of(@_QQclX68) : !fir.ref<!fir.char<1>>
! CHECK:   %[[DESIG2:.*]] = hlfir.designate %[[DECL_CHR]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:   hlfir.assign %{{.*}} to %[[DESIG2]] : !fir.ref<!fir.char<1>>, !fir.boxchar<1>
! CHECK:   cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:   %[[BOX_P1:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:   %[[DECL_P1:.*]]:2 = hlfir.declare %[[BOX_P1]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[FIELD_A:.*]] = hlfir.designate %[[DECL_P1]]#0{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[C1_I32]] to %[[FIELD_A]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:   %[[FIELD_B:.*]] = hlfir.designate %[[DECL_P1]]#0{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[C2_I32]] to %[[FIELD_B]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:   %[[CLASS_P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CHECK:   %[[DECL_P2:.*]]:2 = hlfir.declare %[[CLASS_P2]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>) -> (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>)
! CHECK:   %[[C1_I32_P2:.*]] = arith.constant 1 : i32
! CHECK:   %[[P1_PARENT_A:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:   %[[P2_A:.*]] = hlfir.designate %[[P1_PARENT_A]]{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[C1_I32_P2]] to %[[P2_A]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   %[[C2_I32_P2:.*]] = arith.constant 2 : i32
! CHECK:   %[[P1_PARENT_B:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:   %[[P2_B:.*]] = hlfir.designate %[[P1_PARENT_B]]{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[C2_I32_P2]] to %[[P2_B]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   %[[C3_I32_P2:.*]] = arith.constant 3 : i32
! CHECK:   %[[P2_C:.*]] = hlfir.designate %[[DECL_P2]]#0{"c"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[C3_I32_P2]] to %[[P2_C]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   cf.br ^bb{{.*}}

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type8(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "a"}) {
! CFG: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]]{{.*}} {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.class<!fir.array<?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CFG: %[[TC_I32:.*]] = arith.constant 9 : i8
! CFG: %[[CODE_I32:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> i8
! CFG: %[[CMP_I32:.*]] = arith.cmpi eq, %[[CODE_I32]], %[[TC_I32]] : i8
! CFG: cf.cond_br %[[CMP_I32]], ^bb2, ^bb1
! CFG: ^bb1:
! CFG: %[[TC_F32:.*]] = arith.constant 27 : i8
! CFG: %[[CODE_F32:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> i8
! CFG: %[[CMP_F32:.*]] = arith.cmpi eq, %[[CODE_F32]], %[[TC_F32]] : i8
! CFG: cf.cond_br %[[CMP_F32]], ^bb4, ^bb3
! CFG: ^bb2:
! CFG:   %[[BOX_I32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xi32>>
! CFG:   %[[DECL_I32:.*]]:2 = hlfir.declare %[[BOX_I32]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CFG:   %[[C100:.*]] = arith.constant 100 : i32
! CFG:   hlfir.assign %[[C100]] to %[[DECL_I32]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CFG:   cf.br ^bb12
! CFG: ^bb3:
! CFG: %[[TC_CHR:.*]] = arith.constant 40 : i8
! CFG: %[[CODE_CHR:.*]] = fir.box_typecode %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> i8
! CFG: %[[CMP_CHR:.*]] = arith.cmpi eq, %[[CODE_CHR]], %[[TC_CHR]] : i8
! CFG: cf.cond_br %[[CMP_CHR]], ^bb6, ^bb5
! CFG: ^bb4:
! CFG:   %[[BOX_F32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xf32>>
! CFG:   %[[DECL_F32:.*]]:2 = hlfir.declare %[[BOX_F32]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CFG:   %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
! CFG:   hlfir.assign %[[CST2]] to %[[DECL_F32]]#0 : f32, !fir.box<!fir.array<?xf32>>
! CFG:   cf.br ^bb12
! CFG: ^bb5:
! CFG: %[[TDESC_P1:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[BTDESC_P1:.*]] = fir.box_tdesc %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG: %[[CMP_P1:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG: cf.cond_br %[[CMP_P1]], ^bb8, ^bb7
! CFG: ^bb6:
! CFG:   %[[BOX_CHR:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CFG:   %[[DECL_CHR:.*]]:2 = hlfir.declare %[[BOX_CHR]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CFG:   %[[CHR_C:.*]] = fir.address_of(@_QQclX63) : !fir.ref<!fir.char<1>>
! CFG:   %[[DESIG1:.*]] = hlfir.designate %[[DECL_CHR]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CFG:   hlfir.assign %{{.*}} to %[[DESIG1]] : !fir.ref<!fir.char<1>>, !fir.boxchar<1>
! CFG:   %[[CHR_H:.*]] = fir.address_of(@_QQclX68) : !fir.ref<!fir.char<1>>
! CFG:   %[[DESIG2:.*]] = hlfir.designate %[[DECL_CHR]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CFG:   hlfir.assign %{{.*}} to %[[DESIG2]] : !fir.ref<!fir.char<1>>, !fir.boxchar<1>
! CFG:   cf.br ^bb12
! CFG: ^bb7:
! CFG: %[[TDESC_P2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG: %{{.*}} = fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG: cf.cond_br %{{.*}}, ^bb10, ^bb9
! CFG: ^bb8:
! CFG:   %[[BOX_P1:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG:   %[[DECL_P1:.*]]:2 = hlfir.declare %[[BOX_P1]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CFG:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CFG:   %[[FIELD_A:.*]] = hlfir.designate %[[DECL_P1]]#0{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CFG:   hlfir.assign %[[C1_I32]] to %[[FIELD_A]] : i32, !fir.box<!fir.array<?xi32>>
! CFG:   %[[C2_I32:.*]] = arith.constant 2 : i32
! CFG:   %[[FIELD_B:.*]] = hlfir.designate %[[DECL_P1]]#0{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CFG:   hlfir.assign %[[C2_I32]] to %[[FIELD_B]] : i32, !fir.box<!fir.array<?xi32>>
! CFG:   cf.br ^bb12
! CFG: ^bb9:
! CFG:   cf.br ^bb11
! CFG: ^bb10:
! CFG:   %[[CLASS_P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CFG:   %[[DECL_P2:.*]]:2 = hlfir.declare %[[CLASS_P2]] {uniq_name = "_QMselect_type_lower_testFselect_type8Ea"} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>) -> (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>)
! CFG:   %[[C1_I32_P2:.*]] = arith.constant 1 : i32
! CFG:   %[[P1_PARENT_A:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG:   %[[P2_A:.*]] = hlfir.designate %[[P1_PARENT_A]]{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CFG:   hlfir.assign %[[C1_I32_P2]] to %[[P2_A]] : i32, !fir.box<!fir.array<?xi32>>
! CFG:   %[[C2_I32_P2:.*]] = arith.constant 2 : i32
! CFG:   %[[P1_PARENT_B:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG:   %[[P2_B:.*]] = hlfir.designate %[[P1_PARENT_B]]{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CFG:   hlfir.assign %[[C2_I32_P2]] to %[[P2_B]] : i32, !fir.box<!fir.array<?xi32>>
! CFG:   %[[C3_I32_P2:.*]] = arith.constant 3 : i32
! CFG:   %[[P2_C:.*]] = hlfir.designate %[[DECL_P2]]#0{"c"}   shape %{{.*}} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CFG:   hlfir.assign %[[C3_I32_P2]] to %[[P2_C]] : i32, !fir.box<!fir.array<?xi32>>
! CFG:   cf.br ^bb12
! CFG: ^bb11:
! CFG:   fir.call @_FortranAStopStatementText(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, i1, i1) -> ()
! CFG:   fir.unreachable
! CFG: ^bb12:
! CFG:   return

  subroutine select_type9(a)
    class(p1) :: a(:)

    select type(a)
    type is (p1)
      a%a = 1
      a%b = 2
    type is(p2)
      a%a = 1
      a%b = 2
      a%c = 3
    class default
      stop 'error'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type9(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"}) {
! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type9Ea"} : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CHECK: fir.select_type %[[A]]#1 : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{[0-9]+}}, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb{{[0-9]+}}, unit, ^bb{{[0-9]+}}]
! CHECK: ^bb{{[0-9]+}}:
! CHECK: %[[BOX_P1:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: %[[DECL_P1:.*]]:2 = hlfir.declare %[[BOX_P1]] {uniq_name = "_QMselect_type_lower_testFselect_type9Ea"} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CHECK: %[[C1_A:.*]] = arith.constant 1 : i32
! CHECK: %[[FIELD_A_P1:.*]] = hlfir.designate %[[DECL_P1]]#0{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.assign %[[C1_A]] to %[[FIELD_A_P1]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK: %[[C2_B:.*]] = arith.constant 2 : i32
! CHECK: %[[FIELD_B_P1:.*]] = hlfir.designate %[[DECL_P1]]#0{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.assign %[[C2_B]] to %[[FIELD_B_P1]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK: cf.br ^bb{{[0-9]+}}
! CHECK: ^bb{{[0-9]+}}:
! CHECK: %[[BOX_P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CHECK: %[[DECL_P2:.*]]:2 = hlfir.declare %[[BOX_P2]] {uniq_name = "_QMselect_type_lower_testFselect_type9Ea"} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>)
! CHECK: %[[C1_PA:.*]] = arith.constant 1 : i32
! CHECK: %[[PARENT_A:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: %[[FIELD_A_P2:.*]] = hlfir.designate %[[PARENT_A]]{"a"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.assign %[[C1_PA]] to %[[FIELD_A_P2]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK: %[[C2_PB:.*]] = arith.constant 2 : i32
! CHECK: %[[PARENT_B:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: %[[FIELD_B_P2:.*]] = hlfir.designate %[[PARENT_B]]{"b"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.assign %[[C2_PB]] to %[[FIELD_B_P2]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK: %[[C3_C:.*]] = arith.constant 3 : i32
! CHECK: %[[FIELD_C_P2:.*]] = hlfir.designate %[[DECL_P2]]#0{"c"}   shape %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.assign %[[C3_C]] to %[[FIELD_C_P2]] : i32, !fir.box<!fir.array<?xi32>>
! CHECK: cf.br ^bb{{[0-9]+}}

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type9(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"}) {
! CFG: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type9Ea"}
! CFG: %[[TD1:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[BTD1:.*]] = fir.box_tdesc %[[A]]#1
! CFG: %[[CMP1:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG: cf.cond_br %[[CMP1]], ^bb[[P1_BB:[0-9]+]], ^bb[[NEXT1:[0-9]+]]
! CFG: ^bb[[NEXT1]]:
! CFG: %[[TD2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG: %[[BTD2:.*]] = fir.box_tdesc %[[A]]#1
! CFG: %[[CMP2:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG: cf.cond_br %[[CMP2]], ^bb[[P2_BB:[0-9]+]], ^bb[[DEFAULT_PRE:[0-9]+]]
! CFG: ^bb[[P1_BB]]:
! CFG: %[[BOX_P1:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG: %[[DECL_P1:.*]]:2 = hlfir.declare %[[BOX_P1]]
! CFG: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.box<!fir.array<?xi32>>
! CFG: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.box<!fir.array<?xi32>>
! CFG: cf.br ^bb[[MERGE:[0-9]+]]
! CFG: ^bb[[DEFAULT_PRE]]:
! CFG: cf.br ^bb[[DEFAULT_BODY:[0-9]+]]
! CFG: ^bb[[P2_BB]]:
! CFG: %[[BOX_P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CFG: %[[DECL_P2:.*]]:2 = hlfir.declare %[[BOX_P2]]
! CFG: %[[PA:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}
! CFG: %[[PA_A:.*]] = hlfir.designate %[[PA]]{"a"}
! CFG: hlfir.assign %{{.*}} to %[[PA_A]] : i32, !fir.box<!fir.array<?xi32>>
! CFG: %[[PB:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}
! CFG: %[[PB_B:.*]] = hlfir.designate %[[PB]]{"b"}
! CFG: hlfir.assign %{{.*}} to %[[PB_B]] : i32, !fir.box<!fir.array<?xi32>>
! CFG: %[[PC:.*]] = hlfir.designate %[[DECL_P2]]#0{"c"}
! CFG: hlfir.assign %{{.*}} to %[[PC]] : i32, !fir.box<!fir.array<?xi32>>
! CFG: cf.br ^bb[[MERGE]]
! CFG: ^bb[[DEFAULT_BODY]]:
! CFG: hlfir.declare %[[A]]#1
! CFG: fir.call @_FortranAStopStatementText(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> ()
! CFG: fir.unreachable
! CFG: ^bb[[MERGE]]:
! CFG: return

  subroutine select_type10(a)
    class(p1), pointer :: a
    select type(a)
      type is (p1)
        a%a = 1
      type is (p2)
        a%c = 3
      class is (p1)
        a%a = 5
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type10(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> {fir.bindc_name = "a"}) {
! CHECK:  %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"} : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>)
! CHECK:  %[[SELECTOR:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:  fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb{{.*}}, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, unit, ^bb{{.*}}]
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX_P1:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:  %[[P1_DECL:.*]]:2 = hlfir.declare %[[EXACT_BOX_P1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>)
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[DES_A1:.*]] = hlfir.designate %[[P1_DECL]]#0{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C1]] to %[[DES_A1]] : i32, !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX_P2:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CHECK:  %[[P2_DECL:.*]]:2 = hlfir.declare %[[EXACT_BOX_P2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>)
! CHECK:  %[[C3:.*]] = arith.constant 3 : i32
! CHECK:  %[[DES_C:.*]] = hlfir.designate %[[P2_DECL]]#0{"c"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C3]] to %[[DES_C]] : i32, !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[CL_DECL:.*]]:2 = hlfir.declare %[[SELECTOR]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"} : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>)
! CHECK:  %[[C5:.*]] = arith.constant 5 : i32
! CHECK:  %[[DES_A5:.*]] = hlfir.designate %[[CL_DECL]]#0{"a"}   : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C5]] to %[[DES_A5]] : i32, !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type10(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> {fir.bindc_name = "a"}) {
! CFG:  %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"}
! CFG:  %[[SELECTOR:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG:  %[[TDESC_P1:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:  %[[BTDESC_P1:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:  %[[CMP_P1:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:  cf.cond_br %[[CMP_P1]], ^bb[[P1_BODY:[0-9]+]], ^bb[[P2_TEST:[0-9]+]]
! CFG: ^bb[[P2_TEST]]:
! CFG:  %[[TDESC_P2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG:  %[[BTDESC_P2:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG:  %[[CMP_P2:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:  cf.cond_br %[[CMP_P2]], ^bb[[P2_BODY:[0-9]+]], ^bb[[CLASS_TEST:[0-9]+]]
! CFG: ^bb[[P1_BODY]]:
! CFG:  %[[BA_P1:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:  %[[P1_DECL:.*]]:2 = hlfir.declare %[[BA_P1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"}
! CFG:  %[[C1:.*]] = arith.constant 1 : i32
! CFG:  %[[DES_A1:.*]] = hlfir.designate %[[P1_DECL]]#0{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CFG:  hlfir.assign %[[C1]] to %[[DES_A1]] : i32, !fir.ref<i32>
! CFG:  cf.br ^bb[[MERGE:[0-9]+]]
! CFG: ^bb[[CLASS_TEST]]:
! CFG:  %[[TDESC_CL:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:  %[[TDESC_REF:.*]] = fir.convert %[[TDESC_CL]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CFG:  %[[BOX_NONE:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CFG:  %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_REF]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:  cf.cond_br %[[CLASS_IS]], ^bb[[CLASS_BODY:[0-9]+]], ^bb[[UNIT_BODY:[0-9]+]]
! CFG: ^bb[[P2_BODY]]:
! CFG:  %[[BA_P2:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG:  %[[P2_DECL:.*]]:2 = hlfir.declare %[[BA_P2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"}
! CFG:  %[[C3:.*]] = arith.constant 3 : i32
! CFG:  %[[DES_C:.*]] = hlfir.designate %[[P2_DECL]]#0{"c"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<i32>
! CFG:  hlfir.assign %[[C3]] to %[[DES_C]] : i32, !fir.ref<i32>
! CFG:  cf.br ^bb[[MERGE]]
! CFG: ^bb[[UNIT_BODY]]:
! CFG:  cf.br ^bb[[MERGE]]
! CFG: ^bb[[CLASS_BODY]]:
! CFG:  %[[CL_DECL:.*]]:2 = hlfir.declare %[[SELECTOR]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type10Ea"}
! CFG:  %[[C5:.*]] = arith.constant 5 : i32
! CFG:  %[[DES_A5:.*]] = hlfir.designate %[[CL_DECL]]#0{"a"}   : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CFG:  hlfir.assign %[[C5]] to %[[DES_A5]] : i32, !fir.ref<i32>
! CFG:  cf.br ^bb[[MERGE]]
! CFG: ^bb[[MERGE]]:
! CFG:  return

  subroutine select_type11(a)
    class(p1), allocatable :: a
    select type(a)
      type is (p1)
        a%a = 1
      type is (p2)
        a%a = 2
        a%c = 3
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type11(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> {fir.bindc_name = "a"}) {
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMselect_type_lower_testFselect_type11Ea"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>)
! CHECK: %[[SELECTOR:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb2, unit, ^bb3]
! CHECK: ^bb1:
! CHECK:  %[[EXACT_BOX_P1:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:  %[[DECL_P1:.*]]:2 = hlfir.declare %[[EXACT_BOX_P1]] {uniq_name = "_QMselect_type_lower_testFselect_type11Ea"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>)
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[DES_A:.*]] = hlfir.designate %[[DECL_P1]]#0{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C1]] to %[[DES_A]] : i32, !fir.ref<i32>
! CHECK:  cf.br ^bb3
! CHECK: ^bb2:
! CHECK:  %[[EXACT_BOX_P2:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CHECK:  %[[DECL_P2:.*]]:2 = hlfir.declare %[[EXACT_BOX_P2]] {uniq_name = "_QMselect_type_lower_testFselect_type11Ea"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>)
! CHECK:  %[[C2:.*]] = arith.constant 2 : i32
! CHECK:  %[[DES_P1:.*]] = hlfir.designate %[[DECL_P2]]#0{"p1"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:  %[[DES_P1_A:.*]] = hlfir.designate %[[DES_P1]]{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C2]] to %[[DES_P1_A]] : i32, !fir.ref<i32>
! CHECK:  %[[C3:.*]] = arith.constant 3 : i32
! CHECK:  %[[DES_C:.*]] = hlfir.designate %[[DECL_P2]]#0{"c"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[C3]] to %[[DES_C]] : i32, !fir.ref<i32>
! CHECK:  cf.br ^bb3
! CHECK: ^bb3:
! CHECK:  return

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type11(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> {fir.bindc_name = "a"}) {
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMselect_type_lower_testFselect_type11Ea"}
! CFG: %[[SELECTOR:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG: %[[TDESC_P1:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG: %[[BOX_TDESC_P1:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG: %[[CONV1_P1:.*]] = fir.convert %[[TDESC_P1]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> index
! CFG: %[[CONV2_P1:.*]] = fir.convert %[[BOX_TDESC_P1]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> index
! CFG: %[[CMP_P1:.*]] = arith.cmpi eq, %[[CONV1_P1]], %[[CONV2_P1]] : index
! CFG: cf.cond_br %[[CMP_P1]], ^bb2, ^bb1
! CFG: ^bb1:
! CFG: %[[TDESC_P2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG: %[[BOX_TDESC_P2:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG: %[[CONV1_P2:.*]] = fir.convert %[[TDESC_P2]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> index
! CFG: %[[CONV2_P2:.*]] = fir.convert %[[BOX_TDESC_P2]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> index
! CFG: %[[CMP_P2:.*]] = arith.cmpi eq, %[[CONV1_P2]], %[[CONV2_P2]] : index
! CFG: cf.cond_br %[[CMP_P2]], ^bb4, ^bb3
! CFG: ^bb2:
! CFG: %[[BA_P1:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG: %[[D_P1:.*]]:2 = hlfir.declare %[[BA_P1]] {uniq_name = "_QMselect_type_lower_testFselect_type11Ea"}
! CFG: %[[C1:.*]] = arith.constant 1 : i32
! CFG: %[[DES_A:.*]] = hlfir.designate %[[D_P1]]#0{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CFG: hlfir.assign %[[C1]] to %[[DES_A]] : i32, !fir.ref<i32>
! CFG: cf.br ^bb5
! CFG: ^bb3:
! CFG: cf.br ^bb5
! CFG: ^bb4:
! CFG: %[[BA_P2:.*]] = fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG: %[[D_P2:.*]]:2 = hlfir.declare %[[BA_P2]] {uniq_name = "_QMselect_type_lower_testFselect_type11Ea"}
! CFG: %[[C2:.*]] = arith.constant 2 : i32
! CFG: %[[DES_P1:.*]] = hlfir.designate %[[D_P2]]#0{"p1"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG: %[[DES_P1_A:.*]] = hlfir.designate %[[DES_P1]]{"a"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CFG: hlfir.assign %[[C2]] to %[[DES_P1_A]] : i32, !fir.ref<i32>
! CFG: %[[C3:.*]] = arith.constant 3 : i32
! CFG: %[[DES_C:.*]] = hlfir.designate %[[D_P2]]#0{"c"}   : (!fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<i32>
! CFG: hlfir.assign %[[C3]] to %[[DES_C]] : i32, !fir.ref<i32>
! CFG: cf.br ^bb5
! CFG: ^bb5:
! CFG: return

  subroutine select_type12(a)
    class(p1), pointer :: a(:)
    select type(a)
      type is (p1)
        a%a = 120
      type is (p2)
        a%c = 121
      class is (p1)
        a%a = 122
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type12(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"}) {
! CHECK:  %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"}
! CHECK:  %[[SELECTOR:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK:  %[[LOAD_DIMS:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[LOAD_DIMS]], %[[C0]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK:  fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb2, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb3, unit, ^bb4]
! CHECK: ^bb1:
! CHECK:  %[[P1_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:  %[[P1_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK:  %[[P1_DECL:.*]]:2 = hlfir.declare %[[P1_BOX]](%[[P1_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CHECK:  hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.box<!fir.array<?xi32>>
! CHECK:  cf.br ^bb4
! CHECK: ^bb2:
! CHECK:  %[[P2_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>>
! CHECK:  %[[P2_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK:  %[[P2_DECL:.*]]:2 = hlfir.declare %[[P2_BOX]](%[[P2_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"}
! CHECK:  cf.br ^bb4
! CHECK: ^bb3:
! CHECK:  %[[CLS_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK:  %[[CLS_DECL:.*]]:2 = hlfir.declare %[[SELECTOR]](%[[CLS_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CHECK:  cf.br ^bb4
! CHECK: ^bb4:
! CHECK:  return

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type12(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"}) {
! CFG:  %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"}
! CFG:  %[[SELECTOR:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CFG:  %[[LOAD_DIMS:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CFG:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[LOAD_DIMS]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CFG:  %[[P1_TDESC:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:  %[[P1_BOXTD:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:  %[[P1_CMP:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:  cf.cond_br %[[P1_CMP]], ^bb2, ^bb1
! CFG: ^bb1:
! CFG:  %[[P2_TDESC:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG:  %[[P2_BOXTD:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG:  %[[P2_CMP:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:  cf.cond_br %[[P2_CMP]], ^bb4, ^bb3
! CFG: ^bb2:
! CFG:  %[[P1_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG:  %[[P1_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CFG:  %[[P1_DECL:.*]]:2 = hlfir.declare %[[P1_BOX]](%[[P1_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"}
! CFG:  cf.br ^bb7
! CFG: ^bb3:
! CFG:  %[[CLS_TDESC:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:  %[[CLS_CALL:.*]] = fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:  cf.cond_br %[[CLS_CALL]], ^bb6, ^bb5
! CFG: ^bb4:
! CFG:  %[[P2_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>>
! CFG:  %[[P2_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CFG:  %[[P2_DECL:.*]]:2 = hlfir.declare %[[P2_BOX]](%[[P2_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"}
! CFG:  cf.br ^bb7
! CFG: ^bb5:
! CFG:  cf.br ^bb7
! CFG: ^bb6:
! CFG:  %[[CLS_SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CFG:  %[[CLS_DECL:.*]]:2 = hlfir.declare %[[SELECTOR]](%[[CLS_SHIFT]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMselect_type_lower_testFselect_type12Ea"} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>)
! CFG:  cf.br ^bb7
! CFG: ^bb7:
! CFG:  return

  subroutine select_type13(a)
    class(p1), pointer :: a(:)
    select type (a)
      class default
        print*, 'default'
      class is (p1)
        print*, 'class'
    end select

    select type (a)
      type is (p1)
        print*, 'type'
      class default
        print*, 'default'
      class is (p1)
        print*, 'class'
    end select

  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type13(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"})
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DSCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMselect_type_lower_testFselect_type13Ea"}
! CHECK: fir.select_type %{{.*}} : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb2, unit, ^bb1]
! CHECK: ^bb1:
! CHECK: ^bb2:
! CHECK: ^bb3:
! CHECK: fir.select_type %{{.*}} : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb4, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb6, unit, ^bb5]
! CHECK: ^bb4:
! CHECK: ^bb5:
! CHECK: ^bb6:
! CHECK: ^bb7:

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type13(
! CFG-NOT: fir.select_type
! CFG: %[[CLASSIS1:.*]] = fir.call @_FortranAClassIs(
! CFG: cf.cond_br %[[CLASSIS1]], ^bb3, ^bb2
! CFG: ^bb1:
! CFG: cf.br ^bb4
! CFG: ^bb2:
! CFG: cf.br ^bb1
! CFG: ^bb3:
! CFG: cf.br ^bb4
! CFG: ^bb4:
! CFG: %[[CMP:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG: cf.cond_br %[[CMP]], ^bb6, ^bb5
! CFG: ^bb5:
! CFG: %[[CLASSIS2:.*]] = fir.call @_FortranAClassIs(
! CFG: cf.cond_br %[[CLASSIS2]], ^bb9, ^bb8
! CFG: ^bb6:
! CFG: cf.br ^bb10
! CFG: ^bb7:
! CFG: cf.br ^bb10
! CFG: ^bb8:
! CFG: cf.br ^bb7
! CFG: ^bb9:
! CFG: cf.br ^bb10
! CFG: ^bb10:
! CFG: return

  subroutine select_type14(a, b)
    class(p1) :: a, b

    select type(a)
      type is (p2)
        select type (b)
          type is (p2)
            print*,a%c,b%C
        end select
      class default
        print*,a%a
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type14(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "b"})
! CHECK:  %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CHECK:  %[[B:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QMselect_type_lower_testFselect_type14Eb"}
! CHECK:  fir.select_type %[[A]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[OUTER_TI:[0-9]+]], unit, ^bb[[OUTER_DEFAULT:[0-9]+]]]
! CHECK:  ^bb[[OUTER_TI]]:
! CHECK:    %[[A_ADDR:.*]] = fir.box_addr %[[A]]#1
! CHECK:    %[[A_P2:.*]]:2 = hlfir.declare %[[A_ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CHECK:    fir.select_type %[[B]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[INNER_TI:[0-9]+]], unit, ^bb{{[0-9]+}}]
! CHECK:  ^bb[[INNER_TI]]:
! CHECK:    %[[B_ADDR:.*]] = fir.box_addr %[[B]]#1
! CHECK:    %[[B_P2:.*]]:2 = hlfir.declare %[[B_ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type14Eb"}
! CHECK:    %[[AC:.*]] = hlfir.designate %[[A_P2]]#0{"c"}
! CHECK:    fir.load %[[AC]]
! CHECK:    %[[BC:.*]] = hlfir.designate %[[B_P2]]#0{"c"}
! CHECK:    fir.load %[[BC]]
! CHECK:  ^bb[[OUTER_DEFAULT]]:
! CHECK:    %[[A_DEFAULT:.*]]:2 = hlfir.declare %[[A]]#1 {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CHECK:    %[[AA:.*]] = hlfir.designate %[[A_DEFAULT]]#0{"a"}
! CHECK:    fir.load %[[AA]]
! CHECK:  return

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type14(
! CFG-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"},
! CFG-SAME:    %[[ARG1:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "b"})
! CFG:   %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CFG:   %[[B:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QMselect_type_lower_testFselect_type14Eb"}
! CFG:   %[[OUTER_TDESC:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG:   %[[OUTER_BOX_TDESC:.*]] = fir.box_tdesc %[[A]]#1
! CFG:   %[[OUTER_LHS:.*]] = fir.convert %[[OUTER_TDESC]]
! CFG:   %[[OUTER_RHS:.*]] = fir.convert %[[OUTER_BOX_TDESC]]
! CFG:   %[[OUTER_CMP:.*]] = arith.cmpi eq, %[[OUTER_LHS]], %[[OUTER_RHS]]
! CFG:   cf.cond_br %[[OUTER_CMP]], ^bb[[OUTER_TI:[0-9]+]], ^bb[[OUTER_NEXT:[0-9]+]]
! CFG: ^bb[[OUTER_NEXT]]:
! CFG:   cf.br ^bb[[OUTER_DEFAULT:[0-9]+]]
! CFG: ^bb[[OUTER_TI]]:
! CFG:   %[[A_ADDR:.*]] = fir.box_addr %[[A]]#1
! CFG:   %[[A_P2:.*]]:2 = hlfir.declare %[[A_ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CFG:   %[[INNER_TDESC:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>
! CFG:   %[[INNER_BOX_TDESC:.*]] = fir.box_tdesc %[[B]]#1
! CFG:   %[[INNER_LHS:.*]] = fir.convert %[[INNER_TDESC]]
! CFG:   %[[INNER_RHS:.*]] = fir.convert %[[INNER_BOX_TDESC]]
! CFG:   %[[INNER_CMP:.*]] = arith.cmpi eq, %[[INNER_LHS]], %[[INNER_RHS]]
! CFG:   cf.cond_br %[[INNER_CMP]], ^bb[[INNER_TI:[0-9]+]], ^bb[[INNER_NEXT:[0-9]+]]
! CFG: ^bb[[INNER_NEXT]]:
! CFG:   cf.br ^bb[[INNER_MERGE:[0-9]+]]
! CFG: ^bb[[INNER_TI]]:
! CFG:   %[[B_ADDR:.*]] = fir.box_addr %[[B]]#1
! CFG:   %[[B_P2:.*]]:2 = hlfir.declare %[[B_ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type14Eb"}
! CFG:   %[[AC:.*]] = hlfir.designate %[[A_P2]]#0{"c"}
! CFG:   fir.load %[[AC]]
! CFG:   %[[BC:.*]] = hlfir.designate %[[B_P2]]#0{"c"}
! CFG:   fir.load %[[BC]]
! CFG:   cf.br ^bb[[INNER_MERGE]]
! CFG: ^bb[[INNER_MERGE]]:
! CFG:   cf.br ^bb[[OUTER_MERGE:[0-9]+]]
! CFG: ^bb[[OUTER_DEFAULT]]:
! CFG:   %[[A_DEFAULT:.*]]:2 = hlfir.declare %[[A]]#1 {uniq_name = "_QMselect_type_lower_testFselect_type14Ea"}
! CFG:   %[[AA:.*]] = hlfir.designate %[[A_DEFAULT]]#0{"a"}
! CFG:   fir.load %[[AA]]
! CFG:   cf.br ^bb[[OUTER_MERGE]]
! CFG: ^bb[[OUTER_MERGE]]:
! CFG:   return

  subroutine select_type15(a)
    class(p5) :: a

    select type(x => -a)
      type is (p5)
        print*, x%a
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type15(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>> {fir.bindc_name = "a"}) {
! CHECK: %[[RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {bindc_name = ".result"}
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type15Ea"} : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, !fir.dscope) -> (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>)
! CHECK: %[[TMP_RES:.*]] = fir.dispatch "negate"(%[[A]]#0 : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) (%[[A]]#0 : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {pass_arg_pos = 0 : i32}
! CHECK: fir.save_result %[[TMP_RES]] to %[[RES]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CHECK: %[[LOAD_RES:.*]] = fir.load %[[RES]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CHECK: %[[DECL_RES:.*]]:2 = hlfir.declare %[[LOAD_RES]] {uniq_name = ".tmp.func_result"} : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>) -> (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>)
! CHECK: fir.select_type %[[DECL_RES]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, ^bb1, unit, ^bb2]
! CHECK: ^bb1:
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[DECL_RES]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type15Ex"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> (!fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, !fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>)
! CHECK: hlfir.designate %[[X]]#0{"a"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.ref<i32>

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type15(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>> {fir.bindc_name = "a"}) {
! CFG: %[[RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {bindc_name = ".result"}
! CFG: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg 1 {uniq_name = "_QMselect_type_lower_testFselect_type15Ea"}
! CFG: %[[TDESC_A:.*]] = fir.box_tdesc %[[A]]#0 : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.tdesc<none>
! CFG: %[[CALL:.*]] = fir.call %{{.*}}(%[[A]]#0) : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>
! CFG: fir.save_result %[[CALL]] to %[[RES]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CFG: %[[LOAD_RES:.*]] = fir.load %[[RES]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CFG: %[[DECL_RES:.*]]:2 = hlfir.declare %[[LOAD_RES]] {uniq_name = ".tmp.func_result"}
! CFG: %[[TD_P5:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp5{a:i32}>
! CFG: %[[TD_DYN:.*]] = fir.box_tdesc %[[DECL_RES]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.tdesc<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CFG: %[[CONV1:.*]] = fir.convert %[[TD_P5]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> index
! CFG: %[[CONV2:.*]] = fir.convert %[[TD_DYN]] : (!fir.tdesc<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> index
! CFG: %[[CMP:.*]] = arith.cmpi eq, %[[CONV1]], %[[CONV2]] : index
! CFG: cf.cond_br %[[CMP]], ^bb2, ^bb1
! CFG: ^bb1:
! CFG: cf.br ^bb3
! CFG: ^bb2:
! CFG: %[[ADDR:.*]] = fir.box_addr %[[DECL_RES]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CFG: %[[X:.*]]:2 = hlfir.declare %[[ADDR]] {uniq_name = "_QMselect_type_lower_testFselect_type15Ex"}
! CFG: hlfir.designate %[[X]]#0{"a"} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.ref<i32>
! CFG: cf.br ^bb3
! CFG: ^bb3:
! CFG: %[[LD1:.*]] = fir.load %[[RES]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CFG: %[[BOX_NONE:.*]] = fir.convert %[[LD1]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>) -> !fir.box<none>
! CFG: fir.call @_FortranADestroy(%[[BOX_NONE]]) {{.*}} : (!fir.box<none>) -> ()
! CFG: %[[LD2:.*]] = fir.load %[[RES]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CFG: %[[HEAP:.*]] = fir.box_addr %[[LD2]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>) -> !fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CFG: %[[I64:.*]] = fir.convert %[[HEAP]] : (!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> i64
! CFG: %[[NE:.*]] = arith.cmpi ne, %[[I64]], %{{.*}} : i64
! CFG: fir.if %[[NE]] {
! CFG: fir.freemem %[[HEAP]] : !fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CFG: }
! CFG: return

end module

program test_select_type
  use select_type_lower_test

  integer :: a
  integer :: arr(2)
  real :: b
  real :: barr(2)
  character(1) :: carr(2)
  type(p4) :: t4
  type(p1), target :: t1
  type(p2), target :: t2
  type(p1), target :: t1arr(2)
  type(p2) :: t2arr(2)
  class(p1), pointer :: p
  class(p1), allocatable :: p1alloc
  class(p1), allocatable :: p2alloc
  class(p1), pointer :: parr(:)

  call select_type7(t4)
  call select_type7(t2)
  call select_type7(t1)

  call select_type1(t1)
  call select_type1(t2)
  call select_type1(t4)

  call select_type6(a)
  print*, a

  call select_type6(b)
  print*, b

  print*, '> select_type8 with type(p1), dimension(2)'
  call select_type8(t1arr)
  print*, t1arr(1)
  print*, t1arr(2)

  print*, '> select_type8 with type(p2), dimension(2)'
  call select_type8(t2arr)
  print*, t2arr(1)
  print*, t2arr(2)

  print*, '> select_type8 with integer, dimension(2)'
  call select_type8(arr)
  print*, arr(:)

  print*, '> select_type8 with real, dimension(2)'
  call select_type8(barr)
  print*, barr(:)

  print*, '> select_type8 with character(1), dimension(2)'
  call select_type8(carr)
  print*, carr(:)

  t1%a = 0
  p => t1
  print*, '> select_type10'
  call select_type10(p)
  print*, t1

  t2%c = 0
  p => t2
  print*, '> select_type10'
  call select_type10(p)
  print*, t2

  allocate(p1::p1alloc)
  print*, '> select_type11'
  call select_type11(p1alloc)
  print*, p1alloc%a

  allocate(p2::p2alloc)
  print*, '> select_type11'
  call select_type11(p2alloc)
  print*, p2alloc%a

  parr => t1arr
  call select_type12(parr)
  print*, t1arr(1)
  print*, t1arr(2)
end
