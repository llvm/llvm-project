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
! CHECK: %[[SELECTOR:.*]] = fir.embox %[[COORD]] source_box %[[ARG0_LOAD]] : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME: [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[TYPE_IS_BLK:.*]], #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^[[CLASS_IS_BLK:.*]], unit, ^[[DEFAULT_BLK:.*]]]
! CHECK: ^[[TYPE_IS_BLK]]
! CHECK: ^[[CLASS_IS_BLK]]
! CHECK: ^[[DEFAULT_BLK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CFG-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "a"}) {
! CFG:      %[[SELECTOR:.*]] = fir.embox %{{.*}} source_box %{{.*}} : (!fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, !fir.class<{{.*}}>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
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
    type is (character(*))
      print*, 'type is character'
    class default
      print*,'default'
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"})
! CHECK: fir.select_type %[[ARG0]] : !fir.class<none>
! CHECK-SAME: [#fir.type_is<i8>, ^[[I8_BLK:.*]], #fir.type_is<i32>, ^[[I32_BLK:.*]], #fir.type_is<f32>, ^[[F32_BLK:.*]], #fir.type_is<!fir.logical<4>>, ^[[LOG_BLK:.*]], #fir.type_is<!fir.char<1,?>>, ^[[CHAR_BLK:.*]], unit, ^[[DEFAULT:.*]]] 
! CHECK: ^[[I8_BLK]]
! CHECK: ^[[I32_BLK]]
! CHECK: ^[[F32_BLK]]
! CHECK: ^[[LOG_BLK]]
! CHECK: ^[[CHAR_BLK]]
! CHECK: ^[[DEFAULT_BLOCK]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CFG-SAME: %[[SELECTOR:.*]]: !fir.class<none> {fir.bindc_name = "a"}) {

! CFG:  %[[INT8_TC:.*]] = arith.constant 7 : i8
! CFG:  %[[TYPE_CODE:.*]] = fir.box_typecode %[[SELECTOR]] : (!fir.class<none>) -> i8
! CFG:  %[[IS_INT8:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[INT8_TC]] : i8
! CFG:  cf.cond_br %[[IS_INT8]], ^[[INT8_BLK:.*]], ^[[NOT_INT8:.*]]
! CFG: ^[[NOT_INT8]]:
! CFG:  %[[INT32_TC:.*]] = arith.constant 9 : i8
! CFG:  %[[TYPE_CODE:.*]] = fir.box_typecode %[[SELECTOR]] : (!fir.class<none>) -> i8
! CFG:  %[[IS_INT32:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[INT32_TC]] : i8
! CFG:  cf.cond_br %[[IS_INT32]], ^[[INT32_BLK:.*]], ^[[NOT_INT32_BLK:.*]]
! CFG: ^[[INT8_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK:.*]]
! CFG: ^[[NOT_INT32_BLK]]:
! CFG:  %[[FLOAT_TC:.*]] = arith.constant 27 : i8
! CFG:  %[[TYPE_CODE:.*]] = fir.box_typecode %[[SELECTOR]] : (!fir.class<none>) -> i8
! CFG:  %[[IS_FLOAT:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[FLOAT_TC]] : i8
! CFG:  cf.cond_br %[[IS_FLOAT]], ^[[FLOAT_BLK:.*]], ^[[NOT_FLOAT_BLK:.*]]
! CFG: ^[[INT32_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_FLOAT_BLK]]:
! CFG:  %[[LOGICAL_TC:.*]] = arith.constant 14 : i8
! CFG:  %[[TYPE_CODE:.*]] = fir.box_typecode %[[SELECTOR]] : (!fir.class<none>) -> i8
! CFG:  %[[IS_LOGICAL:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[LOGICAL_TC]] : i8
! CFG:  cf.cond_br %[[IS_LOGICAL]], ^[[LOGICAL_BLK:.*]], ^[[NOT_LOGICAL_BLK:.*]]
! CFG: ^[[FLOAT_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_LOGICAL_BLK]]:
! CFG:  %[[CHAR_TC:.*]] = arith.constant 40 : i8
! CFG:  %[[TYPE_CODE:.*]] = fir.box_typecode %[[SELECTOR]] : (!fir.class<none>) -> i8
! CFG:  %[[IS_CHAR:.*]] = arith.cmpi eq, %[[TYPE_CODE]], %[[CHAR_TC]] : i8
! CFG:  cf.cond_br %[[IS_CHAR]], ^[[CHAR_BLK:.*]], ^[[NOT_CHAR_BLK:.*]]
! CFG: ^[[LOGICAL_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK]]
! CFG: ^[[NOT_CHAR_BLK]]:
! CFG:  cf.br ^[[DEFAULT_BLK:.*]]
! CFG: ^[[CHAR_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK]]
! CFG: ^[[DEFAULT_BLK]]:
! CFG:  cf.br ^[[EXIT_BLK]]
! CFG: ^bb12:
! CFG:  return

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
! CHECK: %[[SELECTOR:.*]] = fir.rebox %[[ARG0]] : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.array<?xnone>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.array<?xnone>> [#fir.type_is<i32>, ^{{.*}}, #fir.type_is<f32>, ^{{.*}}, #fir.type_is<!fir.char<1,?>>, ^bb{{.*}}, unit, ^{{.*}}]
! CHECK: ^bb{{.*}}:
! CHECK:  %[[BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xi32>> 
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[SELECTOR_DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[BOX]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:  %[[C100:.*]] = arith.constant 100 : i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[UB:.*]] = arith.subi %[[SELECTOR_DIMS:.*]]#1, %[[C1]] : index
! CHECK:  %[[LOOP_RES:.*]] = fir.do_loop %[[IND:.*]] = %[[C0:.*]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %[[C100]], %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[LOOP_RES]] to %[[BOX]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:  cf.br ^{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xf32>> 
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[SELECTOR_DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[BOX]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:  %[[VALUE:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[UB:.*]] = arith.subi %[[SELECTOR_DIMS]]#1, %[[C1]] : index
! CHECK:  %[[LOOP_RES:.*]] = fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xf32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %[[VALUE]], %[[IND]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xf32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[LOOP_RES]] to %[[BOX]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK:  cf.br ^{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[BOX:.*]] = fir.convert %0 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>> 
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:  %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_A]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:  %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>
! CHECK:  %[[FIELD_B:.*]] = fir.field_index b, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_B]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32> 
! CHECK:  %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %c{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>
! CHECK:  cf.br ^{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[CLASS_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>
! CHECK:  %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[CLASS_BOX]], %[[C0]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_A]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[CLASS_BOX]] [%[[SLICE]]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:  %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[CLASS_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK:  %[[FIELD_B:.*]] = fir.field_index b, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[CLASS_BOX]], %[[C0]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_B]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[CLASS_BOX]] [%[[SLICE]]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:  %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[CLASS_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK:  %[[FIELD_C:.*]] = fir.field_index c, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[CLASS_BOX]], %[[C0]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_C]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:  %[[ARRAY_LOAD:.*]] = fir.array_load %[[CLASS_BOX]] [%[[SLICE]]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:  %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:    %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:    fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[CLASS_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK:  cf.br ^bb{{.*}}

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
! CHECK: %[[SELECTOR:.*]] = fir.rebox %[[ARG0]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^bb{{.*}}, unit, ^bb{{.*}}]
! CHECK: ^bb{{.*}}:
! CHECK: %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_A]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK: %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>
! CHECK: %[[FIELD_B:.*]] = fir.field_index b, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_B]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK: %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32> 
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %c{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.slice<1>
! CHECK: cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK: %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>> 
! CHECK: %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_A]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK: %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK: %[[FIELD_B:.*]] = fir.field_index b, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_B]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK: %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK: %[[FIELD_C:.*]] = fir.field_index c, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[EXACT_BOX]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[BOX_DIMS]]#1, %[[C1]] path %[[FIELD_C]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK: %[[ARRAY_LOAD:.*]] = fir.array_load %[[EXACT_BOX]] [%[[SLICE]]] : (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[IND:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD]]) -> (!fir.array<?xi32>) {
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %{{.*}}, %[[IND]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD]], %[[DO_RES]] to %[[EXACT_BOX]][%[[SLICE]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.slice<1>
! CHECK: cf.br ^bb{{.*}}

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
! CHECK:  %[[SELECTOR:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:  fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^bb{{.*}}, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb{{.*}}, unit, ^bb{{.*}}]
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK:  %[[COORD_A:.*]] = fir.coordinate_of %[[EXACT_BOX]], %[[FIELD_A]] : (!fir.box<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:  fir.store %[[C1]] to %[[COORD_A]] : !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.ptr<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>
! CHECK:  %[[C3:.*]] = arith.constant 3 : i32
! CHECK:  %[[FIELD_C:.*]] = fir.field_index c, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK:  %[[COORD_C:.*]] = fir.coordinate_of %[[EXACT_BOX]], %[[FIELD_C]] : (!fir.box<!fir.ptr<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:  fir.store %[[C3]] to %[[COORD_C]] : !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}
! CHECK:  %[[C5:.*]] = arith.constant 5 : i32
! CHECK:  %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK:  %[[COORD_A:.*]] = fir.coordinate_of %[[SELECTOR]], %[[FIELD_A]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:  fir.store %[[C5]] to %[[COORD_A]] : !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}

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
! CHECK: %[[SELECTOR:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK: fir.select_type %[[SELECTOR]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^bb2, unit, ^bb3]
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CHECK:  %[[COORD_A:.*]] = fir.coordinate_of %[[EXACT_BOX]], %[[FIELD_A]] : (!fir.box<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:  fir.store %[[C1]] to %[[COORD_A]] : !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.heap<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>
! CHECK:  %[[C3:.*]] = arith.constant 3 : i32
! CHECK:  %[[FIELD_C:.*]] = fir.field_index c, !fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>
! CHECK:  %[[COORD_C:.*]] = fir.coordinate_of %[[EXACT_BOX]], %[[FIELD_C]] : (!fir.box<!fir.heap<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:  fir.store %[[C3]] to %[[COORD_C]] : !fir.ref<i32>
! CHECK:  cf.br ^bb{{.*}}

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
! CHECK:  %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[LOAD]], %[[C0]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> (index, index, index)
! CHECK:  %[[SHIFT:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK:  %[[SELECTOR:.*]] = fir.rebox %[[LOAD]](%[[SHIFT]]) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:  fir.select_type %[[SELECTOR]] : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb1, #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>, ^bb2, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb3, unit, ^bb4]
! CHECK: ^bb{{.*}}:
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK: ^bb{{.*}}:  // pred: ^bb0
! CHECK:  %[[EXACT_BOX:.*]] = fir.convert %[[SELECTOR]] : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{a:i32,b:i32,c:i32}>>>


  ! Test correct lowering when CLASS DEFAULT is not at the last position in the
  ! SELECT TYPE construct.
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

! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type13
! CHECK: fir.select_type %{{.*}} : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb2, unit, ^bb1]
! CHECK: ^bb1:
! CHECK: ^bb2:
! CHECK: ^bb3:
! CHECK: fir.select_type %{{.*}} : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb4, #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb6, unit, ^bb5]
! CHECK: ^bb4:
! CHECK: ^bb5:
! CHECK: ^bb6:
! CHECK: ^bb7:

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

  ! Just makes sure the example can be lowered.
  ! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type14

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
! CHECK: %[[TMP_RES:.*]] = fir.dispatch "negate"(%[[ARG0]] : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) (%[[ARG0]] : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {pass_arg_pos = 0 : i32}
! CHECK: fir.save_result %[[TMP_RES]] to %[[RES]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CHECK: %[[LOAD_RES:.*]] = fir.load %[[RES]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>>
! CHECK: fir.select_type %[[LOAD_RES]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> [#fir.type_is<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, ^bb1, unit, ^bb2]

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
