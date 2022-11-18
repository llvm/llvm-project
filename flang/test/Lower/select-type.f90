! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

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


  subroutine select_type6(a)
    class(*), intent(out) :: a

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

end module


