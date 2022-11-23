! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s
! RUN: bbc -polymorphic-type -emit-fir %s -o - | fir-opt --cfg-conversion | FileCheck --check-prefix=CFG %s
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

contains

  function get_class()
    class(p1), pointer :: get_class
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

! CHECK: fir.select_type %[[ARG0]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_P1_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^[[CLASS_IS_P2_BLK:.*]], unit, ^[[DEFAULT_BLOCK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_P1_BLK]]
! CHECK: ^[[CLASS_IS_P2_BLK]]
! CHECK: %[[P2:.*]] = fir.convert %[[ARG0:.*]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>
! CHECK: %[[FIELD:.*]] = fir.field_index c, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK: %{{.*}} = fir.coordinate_of %[[P2]], %[[FIELD]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK: ^[[DEFAULT_BLOCK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type1(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[BOX_TDESC:.*]] = fir.box_tdesc %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<none>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> index
! CFG:      %[[BOX_TDESC_CONV:.*]] = fir.convert %[[BOX_TDESC]] : (!fir.tdesc<none>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG:    ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG:      %[[TDESC_P2_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p2) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P2_CONV:.*]] = fir.convert %[[TDESC_P2_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG:      %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P2_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS]], ^bb[[CLASS_IS_P2_BLK:.*]], ^[[NOT_CLASS_IS_P2_BLK:.*]]
! CFG:    ^[[TYPE_IS_P1_BLK]]:
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK:[0-9]]]
! CFG:    ^bb[[NOT_CLASS_IS_P1_BLK:[0-9]]]:
! CFG:      cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CFG:    ^bb[[CLASS_IS_P1_BLK:[0-9]]]:
! CFG:      cf.br ^[[END_SELECT_BLK:.*]]
! CFG:    ^[[NOT_CLASS_IS_P2_BLK]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG:      %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS]], ^bb[[CLASS_IS_P1_BLK]], ^bb[[NOT_CLASS_IS_P1_BLK]]
! CFG:    ^bb[[CLASS_IS_P2_BLK]]:
! CFG:      cf.br ^[[END_SELECT_BLK]]
! CFG:    ^bb[[DEFAULT_BLK]]:
! CFG:      cf.br ^[[END_SELECT_BLK]]
! CFG:    ^[[END_SELECT_BLK]]:
! CFG:      return

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
! CHECK: %[[SELECTOR:.*]] = fir.load %[[RESULT]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_BLK]]
! CHECK: ^[[DEFAULT_BLK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type2() {
! CFG:     %[[CLASS_ALLOCA:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {bindc_name = ".result"}
! CFG:     %[[GET_CLASS:.*]] = fir.call @_QMselect_type_lower_testPget_class() {{.*}} : () -> !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CFG:     fir.save_result %[[GET_CLASS]] to %[[CLASS_ALLOCA]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG:     %[[LOAD_CLASS:.*]] = fir.load %[[CLASS_ALLOCA]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CFG:     %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:     %[[CLASS_TDESC:.*]] = fir.box_tdesc %[[LOAD_CLASS]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<none>
! CFG:     %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> index
! CFG:     %[[BOX_TDESC_CONV:.*]] = fir.convert %[[CLASS_TDESC]] : (!fir.tdesc<none>) -> index
! CFG:     %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG:     cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG:   ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG:     %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:     %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:     %[[BOX_NONE:.*]] = fir.convert %[[LOAD_CLASS]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CFG:     %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:     cf.cond_br %[[CLASS_IS]], ^[[CLASS_IS_BLK:.*]], ^[[NOT_CLASS_IS_BLK:.*]]
! CFG:   ^[[TYPE_IS_P1_BLK]]:
! CFG:     cf.br ^bb[[EXIT_SELECT_BLK:[0-9]]]
! CFG:   ^[[NOT_CLASS_IS_BLK]]:
! CFG:     cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CFG:   ^[[CLASS_IS_BLK]]:
! CFG:     cf.br ^bb[[END_SELECT_BLK:[0-9]]]
! CFG:   ^bb[[DEFAULT_BLK]]:
! CFG:     cf.br ^bb[[END_SELECT_BLK:[0-9]]]
! CFG:   ^bb[[END_SELECT_BLK:[0-9]]]:
! CFG:     return

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
! CHECK: %[[ARG0_LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG0_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK: %[[TDESC:.*]] = fir.box_tdesc %[[ARG0_LOAD]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.tdesc<none>
! CHECK: %[[SELECTOR:.*]] = fir.embox %[[COORD]] tdesc %[[TDESC]] : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.tdesc<none>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_BLK]]
! CHECK: ^[[DEFAULT_BLK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"}) {
! CFG:      %[[SELECTOR:.*]] = fir.embox %{{.*}} tdesc %{{.*}} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.tdesc<none>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[SELECTOR_TDESC:.*]] = fir.box_tdesc %[[SELECTOR]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<none>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> index
! CFG:      %[[TDESC_CONV:.*]] = fir.convert %[[SELECTOR_TDESC]] : (!fir.tdesc<none>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P1_CONV]], %[[TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[TYPE_IS_P1_BLK:.*]], ^[[NOT_TYPE_IS_P1_BLK:.*]]
! CFG:    ^[[NOT_TYPE_IS_P1_BLK]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CFG:      %[[CLASS_IS:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS]], ^[[CLASS_IS_BLK:.*]], ^[[NOT_CLASS_IS:.*]]
! CFG:    ^[[TYPE_IS_P1_BLK]]:
! CFG:        cf.br ^bb[[END_SELECT_BLK:[0-9]]]
! CFG:    ^[[NOT_CLASS_IS]]:
! CFG:        cf.br ^bb[[DEFAULT_BLK:[0-9]]]
! CFG:    ^[[CLASS_IS_BLK]]:
! CFG:        cf.br ^bb[[END_SELECT_BLK]]
! CFG:    ^bb[[DEFAULT_BLK]]:
! CFG:        cf.br ^bb[[END_SELECT_BLK]]
! CFG:    ^bb[[END_SELECT_BLK]]:
! CFG:        return

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
! CHECK: fir.select_type %[[ARG0]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp3K8{a:i32,b:i32,r:f64}>>, ^[[P3_8:.*]], #fir.type_is<!fir.type<_QMselect_type_lower_testTp3K4{a:i32,b:i32,r:f32}>>, ^[[P3_4:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[P1:.*]], unit, ^[[EXIT:.*]]]
! CHECK: ^[[P3_8]]
! CHECK: ^[[P3_4]]
! CHECK: ^[[P1]]
! CHECK: ^[[EXIT]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CFG:      %[[TDESC_P3_8_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p3.8) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[BOX_TDESC:.*]] = fir.box_tdesc %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<none>
! CFG:      %[[TDESC_P3_8_CONV:.*]] = fir.convert %[[TDESC_P3_8_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> index
! CFG:      %[[BOX_TDESC_CONV:.*]] = fir.convert %[[BOX_TDESC]] : (!fir.tdesc<none>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P3_8_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[P3_8_BLK:.*]], ^[[NOT_P3_8_BLK:.*]]
! CFG:    ^[[NOT_P3_8_BLK]]:
! CFG:      %[[TDESC_P3_4_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p3.4) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[BOX_TDESC:.*]] = fir.box_tdesc %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.tdesc<none>
! CFG:      %[[TDESC_P3_4_CONV:.*]] = fir.convert %[[TDESC_P3_4_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> index
! CFG:      %[[BOX_TDESC_CONV:.*]] = fir.convert %[[BOX_TDESC]] : (!fir.tdesc<none>) -> index
! CFG:      %[[TDESC_CMP:.*]] = arith.cmpi eq, %[[TDESC_P3_4_CONV]], %[[BOX_TDESC_CONV]] : index
! CFG:      cf.cond_br %[[TDESC_CMP]], ^[[P3_4_BLK:.*]], ^[[NOT_P3_4_BLK:.*]]
! CFG:    ^[[P3_8_BLK]]:
! CFG:      _FortranAioOutputAscii
! CFG:      cf.br ^bb[[EXIT_SELECT_BLK:[0-9]]]
! CFG:    ^[[NOT_P3_4_BLK]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
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
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CHECK: fir.select_type %[[ARG0]] : !fir.class<none>
! CHECK-SAME: [#fir.type_is<i8>, ^[[I8_BLK:.*]], #fir.type_is<i32>, ^[[I32_BLK:.*]], #fir.type_is<f32>, ^[[F32_BLK:.*]], #fir.type_is<!fir.logical<4>>, ^[[LOG_BLK:.*]], unit, ^[[DEFAULT:.*]]] 
! CHECK: ^[[I8_BLK]]
! CHECK: ^[[I32_BLK]]
! CHECK: ^[[F32_BLK]]
! CHECK: ^[[LOG_BLK]]
! CHECK: ^[[DEFAULT_BLOCK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type5(

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

! CHECK: fir.select_type %[[ARG0]] : !fir.class<none> [#fir.type_is<i32>, ^[[INT_BLK:.*]], #fir.type_is<f32>, ^[[REAL_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[INT_BLK]]
! CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<none>) -> !fir.ref<i32>
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  fir.store %[[C100]] to %[[BOX_ADDR]] : !fir.ref<i32>

! CHECK: ^[[REAL_BLK]]:  // pred: ^bb0
! CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<none>) -> !fir.ref<f32>
! CHECK:  %[[C2:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:  fir.store %[[C2]] to %[[BOX_ADDR]] : !fir.ref<f32>


! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CFG:   %[[INT32_TYPECODE:.*]] = arith.constant 9 : i8
! CFG:   %[[ARG0_TYPECODE:.*]] = fir.box_typecode %[[ARG0]] : (!fir.class<none>) -> i8
! CFG:   %[[IS_TYPECODE:.*]] = arith.cmpi eq, %[[ARG0_TYPECODE]], %[[INT32_TYPECODE]] : i8
! CFG:   cf.cond_br %[[IS_TYPECODE]], ^[[TYPE_IS_INT_BLK:.*]], ^[[TYPE_NOT_INT_BLK:.*]]
! CFG: ^[[TYPE_NOT_INT_BLK]]:
! CFG:   %[[FLOAT_TYPECODE:.*]] = arith.constant 27 : i8
! CFG:   %[[ARG0_TYPECODE:.*]] = fir.box_typecode %[[ARG0]] : (!fir.class<none>) -> i8
! CFG:   %[[IS_TYPECODE:.*]] = arith.cmpi eq, %[[ARG0_TYPECODE]], %[[FLOAT_TYPECODE]] : i8
! CFG:   cf.cond_br %[[IS_TYPECODE]], ^[[TYPE_IS_REAL_BLK:.*]], ^[[TYPE_NOT_REAL_BLK:.*]]
! CFG: ^[[TYPE_IS_INT_BLK]]:
! CFG:   %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<none>) -> !fir.ref<i32>
! CFG:   %[[C100:.*]] = arith.constant 100 : i32
! CFG:   fir.store %[[C100]] to %[[BOX_ADDR]] : !fir.ref<i32>
! CFG:   cf.br ^[[EXIT_SELECT_BLK:.*]]
! CFG: ^[[TYPE_NOT_REAL_BLK]]:
! CFG:   cf.br ^[[DEFAULT_BLK:.*]]
! CFG: ^[[TYPE_IS_REAL_BLK]]:
! CFG: %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<none>) -> !fir.ref<f32>
! CFG: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
! CFG: fir.store %[[CST]] to %[[BOX_ADDR]] : !fir.ref<f32>
! CFG: cf.br ^[[EXIT_SELECT_BLK]]
! CFG: ^[[DEFAULT_BLK]]:
! CFG:   fir.call @_FortranAStopStatementText
! CFG:   fir.unreachable
! CFG: ^[[EXIT_SELECT_BLK]]:
! CFG   return

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
! CHECK: fir.select_type %[[ARG0]] :
! CHECK-SAME: !fir.class<none> [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^bb2, #fir.class_is<!fir.type<_QMselect_type_lower_testTp4{a:i32,b:i32,c:i32,d:i32}>>, ^bb3, unit, ^bb4]

! Check correct ordering of class is type guard. The expected flow should be:
!   class is (p4) -> class is (p2) -> class is (p1) -> class default

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CFG-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}) {
! CFG:      %[[TDESC_P4_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p4) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P4_CONV:.*]] = fir.convert %[[TDESC_P4_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P4:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P4_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P4]], ^[[CLASS_IS_P4_BLK:.*]], ^[[CLASS_NOT_P4_BLK:.*]]
! CFG:    ^bb[[CLASS_NOT_P1_BLK:[0-9]]]:
! CFG:      cf.br ^[[CLASS_DEFAULT_BLK:.*]]
! CFG:    ^bb[[CLASS_IS_P1_BLK:[0-9]]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK:.*]]
! CFG:    ^bb[[CLASS_NOT_P2_BLK:[0-9]]]:
! CFG:      %[[TDESC_P1_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P1_CONV:.*]] = fir.convert %[[TDESC_P1_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P1:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P1_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P1]], ^bb[[CLASS_IS_P1_BLK]], ^bb[[CLASS_NOT_P1_BLK]]
! CFG:    ^bb[[CLASS_IS_P2_BLK:[0-9]]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:    ^[[CLASS_NOT_P4_BLK]]:
! CFG:      %[[TDESC_P2_ADDR:.*]] = fir.address_of(@_QMselect_type_lower_testE.dt.p2) : !fir.ref<!fir.type<{{.*}}>>
! CFG:      %[[TDESC_P2_CONV:.*]] = fir.convert %[[TDESC_P2_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CFG:      %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CFG:      %[[CLASS_IS_P2:.*]] = fir.call @_FortranAClassIs(%[[BOX_NONE]], %[[TDESC_P2_CONV]]) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:      cf.cond_br %[[CLASS_IS_P2]], ^bb[[CLASS_IS_P2_BLK]], ^bb[[CLASS_NOT_P2_BLK]]
! CFG:   ^[[CLASS_IS_P4_BLK]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:   ^[[CLASS_DEFAULT_BLK]]:
! CFG:      cf.br ^[[EXIT_SELECT_BLK]]
! CFG:   ^[[EXIT_SELECT_BLK]]:
! CFG:      return

end module

program test_select_type
  use select_type_lower_test

  integer :: a
  real :: b
  type(p4) :: t4
  type(p2) :: t2
  type(p1) :: t1

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

end
