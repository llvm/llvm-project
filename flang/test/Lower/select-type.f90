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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{{{.*}}uniq_name = "_QMselect_type_lower_testFselect_type1Ea"}
! CHECK: fir.select_type %[[ADECL]]#1
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp2

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type1(
! CFG: hlfir.declare %{{.*}}
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp1
! CFG: fir.box_tdesc %{{.*}}
! CFG: fir.call @_FortranAClassIs(%{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<none>) -> i1

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
! CHECK: %[[RESULT:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[FCTCALL:.*]] = fir.call @_QMselect_type_lower_testPget_class()
! CHECK: fir.save_result %[[FCTCALL]] to %{{.*}}
! CHECK: %[[SELECTOR:.*]] = fir.load %{{.*}}#0
! CHECK: fir.select_type %[[SELECTOR]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type2() {
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp1
! CFG: fir.call @_FortranAClassIs

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[LOAD:.*]] = fir.load %[[ADECL]]#0
! CHECK: %[[SELECTOR:.*]] = hlfir.designate %[[LOAD]]
! CHECK: fir.select_type %[[SELECTOR]]

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type3(
! CFG: hlfir.designate
! CFG: fir.call @_FortranAClassIs

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[ADECL]]#1
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp3K8
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp3K4
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp1

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type4(
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp3K8
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp3K4
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp1
! CFG: fir.call @_FortranAClassIs

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none>
! CHECK: fir.select_type %{{.*}} : !fir.class<none>
! CHECK-SAME: #fir.type_is<i8>
! CHECK-SAME: #fir.type_is<i32>
! CHECK-SAME: #fir.type_is<f32>
! CHECK-SAME: #fir.type_is<!fir.logical<4>>
! CHECK-SAME: #fir.type_is<!fir.char<1,?>>

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type5(
! CFG: fir.box_typecode %{{.*}} : (!fir.class<none>) -> i8
! CFG: arith.cmpi eq, %{{.*}}, %{{.*}} : i8

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none>
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[ADECL]]#1 : !fir.class<none>
! CHECK-SAME: #fir.type_is<i32>
! CHECK-SAME: #fir.type_is<f32>

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type6(
! CFG: fir.box_typecode %{{.*}} : (!fir.class<none>) -> i8

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none>
! CHECK: fir.select_type %{{.*}} : !fir.class<none>
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp2
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp4

! Check correct ordering of class is type guard. The expected flow should be:
!   class is (p4) -> class is (p2) -> class is (p1) -> class default

! CFG-LABEL: func.func @_QMselect_type_lower_testPselect_type7(
! CFG: fir.type_desc !fir.type<_QMselect_type_lower_testTp4
! CFG: fir.call @_FortranAClassIs

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?xnone>>
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[ADECL]]#1 : !fir.class<!fir.array<?xnone>>
! CHECK-SAME: #fir.type_is<i32>
! CHECK-SAME: #fir.type_is<f32>
! CHECK-SAME: #fir.type_is<!fir.char<1,?>>
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK-SAME: #fir.class_is<!fir.type<_QMselect_type_lower_testTp2

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: fir.select_type %[[ADECL]]#1
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK-SAME: #fir.type_is<!fir.type<_QMselect_type_lower_testTp2

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[SELECTOR:.*]] = fir.load %[[ADECL]]#0
! CHECK: fir.select_type %[[SELECTOR]]

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[SELECTOR:.*]] = fir.load %[[ADECL]]#0
! CHECK: fir.select_type %[[SELECTOR]]

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMselect_type_lower_testTp1
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[LOAD:.*]] = fir.load %[[ADECL]]#0
! CHECK: fir.select_type %{{.*}}

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
! CHECK: fir.select_type %{{[0-9a-zA-Z_]+}}{{.*}}#fir.class_is<!fir.type<_QMselect_type_lower_testTp1
! CHECK: fir.select_type %{{[0-9a-zA-Z_]+}}{{.*}}#fir.type_is<!fir.type<_QMselect_type_lower_testTp1{{[^>]*}}>>{{.*}}#fir.class_is<!fir.type<_QMselect_type_lower_testTp1

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
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>
! CHECK: %[[RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMselect_type_lower_testTp5{a:i32}>>>
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[TMP_RES:.*]] = fir.dispatch "negate"(%[[ADECL]]#0
! CHECK: fir.save_result %[[TMP_RES]] to %{{.*}}
! CHECK: %[[LOAD_RES:.*]] = fir.load %{{.*}}
! CHECK: fir.select_type %{{.*}}

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
