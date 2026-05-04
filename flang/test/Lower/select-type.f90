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
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{{.*}}uniq_name = "_QMselect_type_lower_testFselect_type1Ea"
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[TYPE_IS_P1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[CLASS_IS_P1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[CLASS_IS_P2:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[DEFAULT:[0-9]+]]]
! TYPE IS (p1) block: box_addr to !fir.ref<p1>, then declare local 'a' as ref<p1>.
! CHECK:       ^bb[[TYPE_IS_P1]]:
! CHECK:         %[[T1_ADDR:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         hlfir.declare %[[T1_ADDR]] {{.*}}uniq_name = "_QMselect_type_lower_testFselect_type1Ea"
! CHECK:         cf.br ^bb[[EXIT:[0-9]+]]
! CLASS IS (p1) block: declare local 'a' directly with the class<p1> selector.
! CHECK:       ^bb[[CLASS_IS_P1]]:
! CHECK:         hlfir.declare %[[A]]#1 {{.*}}uniq_name = "_QMselect_type_lower_testFselect_type1Ea"
! CHECK-NOT:     fir.box_addr
! CHECK:         cf.br ^bb[[EXIT]]
! CLASS IS (p2) block: convert class<p1> to class<p2>, declare 'a' as class<p2>, access %c via hlfir.designate.
! CHECK:       ^bb[[CLASS_IS_P2]]:
! CHECK:         %[[CONV:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CHECK:         %[[A_P2:.*]]:2 = hlfir.declare %[[CONV]]
! CHECK:         hlfir.designate %[[A_P2]]#0{"c"}
! CHECK:         cf.br ^bb[[EXIT]]
! Default block: declare 'a' with the original class<p1> selector.
! CHECK:       ^bb[[DEFAULT]]:
! CHECK:         hlfir.declare %[[A]]#1 {{.*}}uniq_name = "_QMselect_type_lower_testFselect_type1Ea"
! CHECK:         cf.br ^bb[[EXIT]]
! CHECK:       ^bb[[EXIT]]:
! CHECK:         return

! In the CFG version, fir.select_type is decomposed into a chain of
! type_desc/box_tdesc/cmpi (for type_is) and _FortranAClassIs (for class_is)
! checks chained through cf.cond_br. Bodies of each arm live in their own
! basic blocks and all branch to a single common exit block.
! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type1(
! CFG:         %[[A:.*]]:2 = hlfir.declare %{{.*}}
! Entry block: type_is(p1) check via tdesc equality, then branch to type_is body or next check.
! CFG:         %[[TDESC_P1:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         %[[BOX_TDESC:.*]] = fir.box_tdesc %[[A]]#1
! CFG:         %[[CMP:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:         cf.cond_br %[[CMP]], ^bb[[TYPE_IS_P1:[0-9]+]], ^bb[[TEST_C2:[0-9]+]]
! Block testing class_is(p2) via _FortranAClassIs, branches to class_is(p2) body or to next test.
! CFG:       ^bb[[TEST_C2]]:
! CFG:         %[[TDESC_P2:.*]] = fir.type_desc !fir.type<_QMselect_type_lower_testTp2
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:         cf.cond_br %{{.*}}, ^bb[[CLASS_IS_P2:[0-9]+]], ^bb[[TEST_C1:[0-9]+]]
! type_is(p1) body: box_addr to ref<p1>, declare 'a' as ref<p1>, branch to exit.
! CFG:       ^bb[[TYPE_IS_P1]]:
! CFG:         fir.box_addr %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CFG:         hlfir.declare %{{.*}}{{.*}}"_QMselect_type_lower_testFselect_type1Ea"
! CFG:         cf.br ^bb[[EXIT:[0-9]+]]
! class_is(p1) body: declare 'a' from class<p1> selector, branch to exit.
! CFG:         hlfir.declare %[[A]]#1 {{.*}}"_QMselect_type_lower_testFselect_type1Ea"
! CFG:         cf.br ^bb[[EXIT]]
! Block testing class_is(p1) via _FortranAClassIs, branches to class_is(p1) body or to default.
! CFG:       ^bb[[TEST_C1]]:
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1
! CFG:         cf.cond_br %{{.*}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
! class_is(p2) body: convert class<p1> -> class<p2>, declare 'a', access %c.
! CFG:       ^bb[[CLASS_IS_P2]]:
! CFG:         fir.convert %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CFG:         hlfir.declare %{{.*}}"_QMselect_type_lower_testFselect_type1Ea"
! CFG:         hlfir.designate %{{.*}}{"c"}
! CFG:         cf.br ^bb[[EXIT]]
! Default body: declare 'a' from class<p1> selector, branch to exit.
! CFG:         hlfir.declare %[[A]]#1 {{.*}}"_QMselect_type_lower_testFselect_type1Ea"
! CFG:         cf.br ^bb[[EXIT]]
! Common exit block: return.
! CFG:       ^bb[[EXIT]]:
! CFG:         return

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

! Selector is the result of get_class() loaded from a fir.alloca temp.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type2()
! CHECK:         %[[RESULT:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>> {bindc_name = ".result"}
! CHECK:         %[[FCTCALL:.*]] = fir.call @_QMselect_type_lower_testPget_class() {{.*}} -> !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:         fir.save_result %[[FCTCALL]] to %{{.*}}
! CHECK:         %[[SELECTOR:.*]] = fir.load %{{.*}}#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:         fir.select_type %[[SELECTOR]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! TYPE IS (p1): box_addr from class<ptr<p1>> to !fir.ref<p1>.
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         hlfir.declare %{{.*}}{{.*}}"_QMselect_type_lower_testFselect_type2Ea"
! CHECK:         cf.br ^bb[[X:[0-9]+]]
! CLASS IS (p1): no box_addr, declare with the loaded class<ptr<p1>> selector itself.
! CHECK:       ^bb[[C1]]:
! CHECK:         hlfir.declare %[[SELECTOR]] {{.*}}"_QMselect_type_lower_testFselect_type2Ea"
! CHECK:         cf.br ^bb[[X]]
! CHECK:       ^bb[[D]]:
! CHECK:         hlfir.declare %[[SELECTOR]] {{.*}}"_QMselect_type_lower_testFselect_type2Ea"
! CHECK:         cf.br ^bb[[X]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type2() {
! CFG:         fir.call @_QMselect_type_lower_testPget_class()
! CFG:         %[[LOAD:.*]] = fir.load %{{.*}}
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         fir.box_tdesc %[[LOAD]]
! CFG:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:         cf.cond_br %{{.*}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1

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

! Selector is a single element designated from an array pointer.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[A_LD:.*]] = fir.load %[[A]]#0
! CHECK:         %[[SELECTOR:.*]] = hlfir.designate %[[A_LD]] (%c1{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>, index) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         fir.select_type %[[SELECTOR]] : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.box_addr %[[SELECTOR]] : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         hlfir.declare %{{.*}}{{.*}}"_QMselect_type_lower_testFselect_type3Ex"
! CHECK:       ^bb[[C1]]:
! CHECK:         hlfir.declare %[[SELECTOR]] {{.*}}"_QMselect_type_lower_testFselect_type3Ex"

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CFG:         hlfir.designate
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         fir.box_tdesc
! CFG:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:         fir.call @_FortranAClassIs

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

! Two type_is alternatives for the parameterised type p3, plus class_is(p1).
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp3K8{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,r:f64}>>, ^bb[[T8:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp3K4{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,r:f32}>>, ^bb[[T4:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[X:[0-9]+]]]
! CHECK:       ^bb[[T8]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp3K8{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,r:f64}>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[T4]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp3K4{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,r:f32}>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[C1]]:
! CHECK:         hlfir.declare %[[A]]#1
! CHECK:       ^bb[[X]]:
! CHECK:         return

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp3K8
! CFG:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:         cf.cond_br
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp3K4
! CFG:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
! CFG:         cf.cond_br
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         fir.call @_FortranAClassIs

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

! Unlimited polymorphic selector: each type_is branch box_addrs to the
! corresponding intrinsic type and declares 'x' with that type. The character
! branch additionally extracts the element size with fir.box_elesize.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<none>
! CHECK-SAME:    [#fir.type_is<i8>, ^bb[[I8:[0-9]+]],
! CHECK-SAME:     #fir.type_is<i32>, ^bb[[I32:[0-9]+]],
! CHECK-SAME:     #fir.type_is<f32>, ^bb[[F32:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.logical<4>>, ^bb[[L4:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.char<1,?>>, ^bb[[CH:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[I8]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<i8>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[I32]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<i32>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[F32]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<f32>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[L4]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<!fir.logical<4>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[CH]]:
! CHECK:         fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         fir.box_elesize %[[A]]#1 : (!fir.class<none>) -> index
! CHECK:         hlfir.declare %{{.*}} typeparams %{{.*}} {{.*}}"_QMselect_type_lower_testFselect_type5Ex"
! CHECK:       ^bb[[D]]:
! CHECK:         hlfir.declare %[[A]]#1

! CFG version uses fir.box_typecode + arith.cmpi against the intrinsic
! type code constants for each type_is alternative.
! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CFG:         fir.box_typecode %{{.*}} : (!fir.class<none>) -> i8
! CFG:         arith.cmpi eq, %{{.*}}, %{{.*}} : i8
! CFG:         cf.cond_br
! CFG:         fir.box_typecode
! CFG:         cf.cond_br

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

! type_is(integer) and type_is(real) bodies do hlfir.assign of a constant
! into the box-addr-typed reference; class default body calls
! _FortranAStopStatementText and ends in fir.unreachable.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<none>
! CHECK-SAME:    [#fir.type_is<i32>, ^bb[[I32:[0-9]+]],
! CHECK-SAME:     #fir.type_is<f32>, ^bb[[F32:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[I32]]:
! CHECK:         %[[I_REF:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<i32>
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]]
! CHECK:         %[[C100:.*]] = arith.constant 100 : i32
! CHECK:         hlfir.assign %[[C100]] to %[[I_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:       ^bb[[F32]]:
! CHECK:         %[[F_REF:.*]] = fir.box_addr %[[A]]#1 : (!fir.class<none>) -> !fir.ref<f32>
! CHECK:         %[[F_DECL:.*]]:2 = hlfir.declare %[[F_REF]]
! CHECK:         %[[C2:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:         hlfir.assign %[[C2]] to %[[F_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:       ^bb[[D]]:
! CHECK:         hlfir.declare %[[A]]#1
! CHECK:         fir.call @_FortranAStopStatementText(
! CHECK:         fir.unreachable

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CFG:         %[[INT_TC:.*]] = arith.constant 9 : i8
! CFG:         %[[A_TC:.*]] = fir.box_typecode %{{.*}}
! CFG:         arith.cmpi eq, %[[A_TC]], %[[INT_TC]] : i8
! CFG:         cf.cond_br
! CFG:         %[[REAL_TC:.*]] = arith.constant 27 : i8

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

! intent(out) class(*) triggers a Destroy + Initialize pair before the
! select_type. Each class_is branch fir.converts the unlimited-polymorphic
! selector to the specific class<T> and declares 'a' with that type.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.call @_FortranADestroy(
! CHECK:         fir.call @_FortranAInitialize(
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<none>
! CHECK-SAME:    [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[C2:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp4{p2:!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>,d:i32}>>, ^bb[[C4:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[C1]]:
! CHECK:         fir.convert %[[A]]#1 : (!fir.class<none>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[C2]]:
! CHECK:         fir.convert %[[A]]#1 : (!fir.class<none>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[C4]]:
! CHECK:         fir.convert %[[A]]#1 : (!fir.class<none>) -> !fir.class<!fir.type<_QMselect_type_lower_testTp4{p2:!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>,d:i32}>>
! CHECK:         hlfir.declare
! CHECK:       ^bb[[D]]:
! CHECK:         hlfir.declare %[[A]]#1

! Check correct ordering of class is type guard. The expected flow should be:
!   class is (p4) -> class is (p2) -> class is (p1) -> class default
! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp4
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}})
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp2
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}})
! CFG:         fir.type_desc !fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>
! CFG:         fir.call @_FortranAClassIs(%{{.*}}, %{{.*}})

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

! Unlimited polymorphic *array* selector: each branch fir.converts to the
! concrete typed box, then declares 'a' with the array shape. Whole-array
! assignments lower to hlfir.assign on the typed box.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type8(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<?xnone>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<!fir.array<?xnone>>
! CHECK-SAME:    [#fir.type_is<i32>, ^bb[[I32:[0-9]+]],
! CHECK-SAME:     #fir.type_is<f32>, ^bb[[F32:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.char<1,?>>, ^bb[[CH:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[C2:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! type_is(integer) -> assign 100 to integer array.
! CHECK:       ^bb[[I32]]:
! CHECK:         %[[A_I32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         hlfir.declare %[[A_I32]]
! CHECK:         %[[C100:.*]] = arith.constant 100 : i32
! CHECK:         hlfir.assign %[[C100]] to %{{.*}}
! type_is(real) -> assign 2.0 to f32 array.
! CHECK:       ^bb[[F32]]:
! CHECK:         %[[A_F32:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         hlfir.declare %[[A_F32]]
! CHECK:         hlfir.assign %{{.*}} to %{{.*}}
! type_is(character(*)) -> two-element character stores via designated
! single-character refs; box_elesize provides the typeparam for each
! hlfir.designate.
! CHECK:       ^bb[[CH]]:
! CHECK:         %[[A_CH:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         hlfir.declare %[[A_CH]]
! CHECK:         fir.box_elesize %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:         hlfir.designate %{{.*}} (%c1{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:         hlfir.assign %{{.*}} to %{{.*}}
! CHECK:         fir.box_elesize %{{.*}}
! CHECK:         hlfir.designate %{{.*}} (%c2{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:         hlfir.assign %{{.*}} to %{{.*}}
! type_is(p1) -> hlfir.designate of {"a"}/{"b"} returns array references.
! CHECK:       ^bb[[T1]]:
! CHECK:         %[[A_P1:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:         hlfir.declare %[[A_P1]]
! CHECK:         hlfir.designate %{{.*}}{"a"} {{.*}}: (!fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         hlfir.assign %c1{{.*}} to %{{.*}}
! CHECK:         hlfir.designate %{{.*}}{"b"}
! CHECK:         hlfir.assign %c2{{.*}} to %{{.*}}
! class_is(p2) -> array of class<p2>, designates {"a"}/{"b"}/{"c"}.
! CHECK:       ^bb[[C2]]:
! CHECK:         %[[A_P2:.*]] = fir.convert %[[A]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CHECK:         hlfir.declare %[[A_P2]]
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:         hlfir.designate %{{.*}}{"b"}
! CHECK:         hlfir.designate %{{.*}}{"c"}

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

! class(p1) :: a(:) selector: each typed branch fir.converts and declares.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type9(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.select_type %[[A]]#1 : !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[T2:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK:         hlfir.declare
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:         hlfir.designate %{{.*}}{"b"}
! CHECK:       ^bb[[T2]]:
! CHECK:         fir.convert %[[A]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>
! CHECK:         hlfir.declare
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:         hlfir.designate %{{.*}}{"b"}
! CHECK:         hlfir.designate %{{.*}}{"c"}

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

! Pointer selector: load the box once and select_type on the loaded class.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type10(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[SEL:.*]] = fir.load %[[A]]#0
! CHECK:         fir.select_type %[[SEL]] : !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[T2:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.box_addr %[[SEL]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:         hlfir.assign %c1{{.*}} to %{{.*}}
! CHECK:       ^bb[[T2]]:
! CHECK:         fir.box_addr %[[SEL]] : (!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>
! CHECK:         hlfir.designate %{{.*}}{"c"}
! CHECK:         hlfir.assign %c3{{.*}} to %{{.*}}
! CHECK:       ^bb[[C1]]:
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:         hlfir.assign %c5{{.*}} to %{{.*}}

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

! Allocatable selector: same load-then-select pattern, with !fir.heap inside
! the selector class.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type11(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[SEL:.*]] = fir.load %[[A]]#0
! CHECK:         fir.select_type %[[SEL]] : !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[T2:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.box_addr %[[SEL]]
! CHECK:         hlfir.designate %{{.*}}{"a"}
! CHECK:       ^bb[[T2]]:
! CHECK:         fir.box_addr %[[SEL]]
! Access to a%a goes through a%p1%a (parent component then field).
! CHECK:         %[[P1:.*]] = hlfir.designate %{{.*}}{"p1"}
! CHECK:         hlfir.designate %[[P1]]{"a"}
! CHECK:         hlfir.designate %{{.*}}{"c"}

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

! Array pointer selector: load the descriptor for the select_type, then
! take the lower bound from box_dims for use as the per-branch fir.shift.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type12(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[A_LD:.*]] = fir.load %[[A]]#0
! CHECK:         %[[A_LD2:.*]] = fir.load %[[A]]#0
! CHECK:         %[[DIMS:.*]]:3 = fir.box_dims %[[A_LD2]], %c0{{.*}}
! CHECK:         fir.select_type %[[A_LD]] : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1:[0-9]+]],
! CHECK-SAME:     #fir.type_is<!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>, ^bb[[T2:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! type_is(p1) -> convert from class<ptr<p1>> to box<ptr<p1>>, shift to set
! the lower bound, declare 'a' as a non-pointer box.
! CHECK:       ^bb[[T1]]:
! CHECK:         fir.convert %[[A_LD]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK:         fir.shift %[[DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK:         hlfir.declare
! CHECK:         hlfir.designate %{{.*}}{"a"}{{.*}}-> !fir.box<!fir.array<?xi32>>
! type_is(p2) -> convert to box<ptr<p2>>, shift, declare, designate {"c"}.
! CHECK:       ^bb[[T2]]:
! CHECK:         fir.convert %[[A_LD]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp2{p1:!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>,c:i32}>>>>
! CHECK:         fir.shift %[[DIMS]]#0
! CHECK:         hlfir.declare
! CHECK:         hlfir.designate %{{.*}}{"c"}
! class_is(p1) -> no convert, shift, declare with the class<ptr<p1>>
! selector and designate {"a"}.
! CHECK:       ^bb[[C1]]:
! CHECK:         fir.shift %[[DIMS]]#0
! CHECK:         hlfir.declare %[[A_LD]](%{{.*}})
! CHECK:         hlfir.designate %{{.*}}{"a"}

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

! Both select_types: even when CLASS DEFAULT is written in the middle of
! the construct, fir.select_type still emits the default as the trailing
! `unit` arm. Each select_type ends with a single common exit block.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type13
! First select: class is (p1), unit (default).
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK-SAME:    [#fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1A:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[DA:[0-9]+]]]
! CHECK:       ^bb[[DA]]:
! CHECK:         cf.br ^bb[[X1:[0-9]+]]
! CHECK:       ^bb[[C1A]]:
! CHECK:         cf.br ^bb[[X1]]
! CHECK:       ^bb[[X1]]:
! Second select: type_is(p1), class_is(p1), unit (default).
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[T1B:[0-9]+]],
! CHECK-SAME:     #fir.class_is<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>, ^bb[[C1B:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[DB:[0-9]+]]]

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

! Nested select_type. Outer has type_is(p2) and unit. Inner inside the
! type_is(p2) block has a nested type_is(p2)/unit pair.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type14
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp2
! CHECK-SAME:     unit
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.type<_QMselect_type_lower_testTp1{a:i32,b:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp2
! CHECK-SAME:     unit

  subroutine select_type15(a)
    class(p5) :: a

    select type(x => -a)
      type is (p5)
        print*, x%a
    end select
  end subroutine

! Selector is the result of an overloaded unary minus implemented as the
! type-bound function negate. The result is a class<heap<p5>> allocated
! by the dispatched function and stored into a fir.alloca .result tmp.
! CHECK-LABEL: func.func @_QMselect_type_lower_testPselect_type15(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CHECK:         %[[RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {bindc_name = ".result"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[CALL:.*]] = fir.dispatch "negate"(%[[A]]#0 : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) (%[[A]]#0 : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>> {pass_arg_pos = 0 : i32}
! CHECK:         fir.save_result %[[CALL]] to %[[RES]]
! CHECK:         %[[LOAD:.*]] = fir.load %{{.*}}
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CHECK-SAME:    [#fir.type_is<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>, ^bb[[T:[0-9]+]],
! CHECK-SAME:     unit, ^bb[[D:[0-9]+]]]
! CHECK:       ^bb[[T]]:
! CHECK:         fir.box_addr %{{.*}} : (!fir.class<{{.*}}>) -> !fir.ref<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CHECK:         hlfir.declare %{{.*}}{{.*}}"_QMselect_type_lower_testFselect_type15Ex"
! CHECK:         hlfir.designate %{{.*}}{"a"}

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
