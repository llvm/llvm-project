! RUN: bbc -emit-hlfir -I nw %s -o - | FileCheck %s

module components_test
  type t1
     integer :: i(6)
     real :: r(5)
  end type t1

  type t2
     type(t1) :: g1(3,3), g2(4,4,4)
     integer :: g3(5)
  end type t2

  type t3
     type(t1) :: h1(3)
     type(t2) :: h2(4)
  end type t3

  type(t3) :: instance

contains

  subroutine s1(i,j)
     i = instance%h2(2)%g2(1,2,3)%i(j)
  end subroutine s1
! CHECK-LABEL:   func.func @_QMcomponents_testPs1(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "j"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QMcomponents_testEinstance) : !fir.ref<!fir.type<_QMcomponents_testTt3
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QMcomponents_testEinstance"} : (!fir.ref<!fir.type<_QMcomponents_testTt3
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QMcomponents_testFs1Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QMcomponents_testFs1Ej"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0{"h2"} <%[[VAL_8]]> (%[[VAL_9]])  : (!fir.ref<!fir.type<_QMcomponents_testTt3
! CHECK:           %[[VAL_11:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_17:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_10]]{"g2"} <%[[VAL_14]]> (%[[VAL_15]], %[[VAL_16]], %[[VAL_17]])  : (!fir.ref<!fir.type<_QMcomponents_testTt2
! CHECK:           %[[VAL_19:.*]] = arith.constant 6 : index
! CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_19]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = hlfir.designate %[[VAL_18]]{"i"} <%[[VAL_20]]> (%[[VAL_22]])  : (!fir.ref<!fir.type<_QMcomponents_testTt1
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

end module components_test


subroutine sliced_base()
  interface
    subroutine takes_int_array(i)
      integer :: i(:)
    end subroutine
  end interface
  type t
    real :: x
    integer :: y
  end type
  type(t) :: a(100)
  a(1:50)%y = 42
  call takes_int_array(a(1:50)%y)
end subroutine
! CHECK-LABEL:   func.func @_QPsliced_base() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<100x!fir.type<_QFsliced_baseTt
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFsliced_baseEa"} : (!fir.ref<!fir.array<100x!fir.type<_QFsliced_baseTt
! CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 50 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 50 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_5]]:%[[VAL_6]]:%[[VAL_7]])  shape %[[VAL_9]] : (!fir.ref<!fir.array<100x!fir.type<_QFsliced_baseTt
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"y"}   shape %[[VAL_9]] : (!fir.ref<!fir.array<50x!fir.type<_QFsliced_baseTt
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_11]] : i32, !fir.box<!fir.array<50xi32>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 50 : index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 50 : index
! CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_12]]:%[[VAL_13]]:%[[VAL_14]])  shape %[[VAL_16]] : (!fir.ref<!fir.array<100x!fir.type<_QFsliced_baseTt
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_17]]{"y"}   shape %[[VAL_16]] : (!fir.ref<!fir.array<50x!fir.type<_QFsliced_baseTt
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.box<!fir.array<50xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QPtakes_int_array(%[[VAL_19]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine issue772(a, x)
  ! Verify that sub-expressions inside a component reference are
  ! only evaluated once.
  type t
    real :: b(100)
  end type
  real :: x(100)
  type(t) :: a(100)
  x = a(ifoo())%b(1:100:1)
  print *, a(20)%b(1:ibar():1)
end subroutine
! CHECK-LABEL:   func.func @_QPissue772(
! CHECK: fir.call @_QPifoo()
! CHECK-NOT: fir.call @_QPifoo()
! CHECK: fir.call @_QPibar()
! CHECK-NOT: fir.call @_QPibar()
! CHECK: return

! -----------------------------------------------------------------------------
!     Test array%character array sections
! -----------------------------------------------------------------------------

subroutine lhs_char_section(a)
  type t
   character(5) :: c
  end type
  type(t) :: a(10)
  a%c = "hello"
end subroutine
! CHECK-LABEL:   func.func @_QPlhs_char_section(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFlhs_char_sectionTt
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) dummy_scope %[[VAL_1]] {uniq_name = "_QFlhs_char_sectionEa"} : (!fir.ref<!fir.array<10x!fir.type<_QFlhs_char_sectionTt
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0{"c"}   shape %[[VAL_3]] typeparams %[[VAL_8]] : (!fir.ref<!fir.array<10x!fir.type<_QFlhs_char_sectionTt
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_9]] : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:           return
! CHECK:         }

subroutine rhs_char_section(a, c)
  type t
   character(10) :: c
  end type
  type(t) :: a(10)
  character(10) :: c(10)
  c = a%c
end subroutine
! CHECK-LABEL:   func.func @_QPrhs_char_section(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFrhs_char_sectionTt
! CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFrhs_char_sectionEa"} : (!fir.ref<!fir.array<10x!fir.type<_QFrhs_char_sectionTt
! CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_10]]) typeparams %[[VAL_8]] dummy_scope %[[VAL_2]] {uniq_name = "_QFrhs_char_sectionEc"} : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.ref<!fir.array<10x!fir.char<1,10>>>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_13:.*]] = hlfir.designate %[[VAL_5]]#0{"c"}   shape %[[VAL_4]] typeparams %[[VAL_12]] : (!fir.ref<!fir.array<10x!fir.type<_QFrhs_char_sectionTt
! CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_11]]#0 : !fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:           return
! CHECK:         }

subroutine elemental_char_section(a, i)
  type t
   character(10) :: c
  end type
  type(t) :: a(10)
  integer :: i(10)
  i = scan(a%c, "hello")
end subroutine
! CHECK-LABEL:   func.func @_QPelemental_char_section(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFelemental_char_sectionTt
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "i"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFelemental_char_sectionEa"} : (!fir.ref<!fir.array<10x!fir.type<_QFelemental_char_sectionTt
! CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFelemental_char_sectionEi"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0{"c"}   shape %[[VAL_4]] typeparams %[[VAL_9]] : (!fir.ref<!fir.array<10x!fir.type<_QFelemental_char_sectionTt
! CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %[[VAL_12]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_10]] (%[[VAL_15]])  typeparams %[[VAL_9]] : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:             %[[VAL_17:.*]] = arith.constant false
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_13]]#0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
! CHECK:             %[[VAL_22:.*]] = fir.call @_FortranAScan1(%[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> i32
! CHECK:             hlfir.yield_element %[[VAL_23]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_8]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<10xi32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPextended_type_components() {
subroutine extended_type_components
  type t1
    integer :: t1i
  end type t1
  type, extends(t1) :: t2
    integer :: t2i
  end type t2
  type, extends(t2) :: t3
    integer :: t3i
  end type t3
  type, extends(t3) :: t4
    integer :: t4i
  end type t4
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<5xi32>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<5xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<5xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<5xi32>>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFextended_type_componentsEu3v"} : (!fir.ref<!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<5x!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = "_QFextended_type_componentsEu3va"} : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3

  type u1
    integer :: u1i
  end type u1
  type, extends(u1) :: u2
    integer :: u2i
    type(t3) :: u2t3
    type(t3) :: u2t4
  end type u2
  type, extends(u2) :: u3
    integer :: u3i
  end type u3

  type(u3) :: u3v
  type(u3) :: u3va(5)

  call foo1(u3v%u2t3%t1i)
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0{"u2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"u2t3"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_12:.*]] = hlfir.designate %[[VAL_11]]{"t2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_13:.*]] = hlfir.designate %[[VAL_12]]{"t1"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_14:.*]] = hlfir.designate %[[VAL_13]]{"t1i"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt1
! CHECK:           fir.call @_QPfoo1(%[[VAL_14]]) fastmath<contract> : (!fir.ref<i32>) -> ()

  call foo2(u3v%u2%u2t3%t2%t1%t1i) ! different syntax for the previous value
! CHECK:           %[[VAL_15:.*]] = hlfir.designate %[[VAL_5]]#0{"u2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_15]]{"u2t3"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_16]]{"t2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_17]]{"t1"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_19:.*]] = hlfir.designate %[[VAL_18]]{"t1i"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt1
! CHECK:           fir.call @_QPfoo2(%[[VAL_19]]) fastmath<contract> : (!fir.ref<i32>) -> ()

  call foo3(u3v%u2t4%t1i)
! CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_5]]#0{"u2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_21:.*]] = hlfir.designate %[[VAL_20]]{"u2t4"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_22:.*]] = hlfir.designate %[[VAL_21]]{"t2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_23:.*]] = hlfir.designate %[[VAL_22]]{"t1"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_24:.*]] = hlfir.designate %[[VAL_23]]{"t1i"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt1
! CHECK:           fir.call @_QPfoo3(%[[VAL_24]]) fastmath<contract> : (!fir.ref<i32>) -> ()

  call foo4(u3v%u2t4%t2i)
! CHECK:           %[[VAL_25:.*]] = hlfir.designate %[[VAL_5]]#0{"u2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_26:.*]] = hlfir.designate %[[VAL_25]]{"u2t4"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_27:.*]] = hlfir.designate %[[VAL_26]]{"t2"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_28:.*]] = hlfir.designate %[[VAL_27]]{"t2i"}   : (!fir.ref<!fir.type<_QFextended_type_componentsTt2
! CHECK:           fir.call @_QPfoo4(%[[VAL_28]]) fastmath<contract> : (!fir.ref<i32>) -> ()

  call foo5(u3va%u2t3%t1i)
! CHECK:           %[[VAL_29:.*]] = hlfir.designate %[[VAL_9]]#0{"u2"}   shape %[[VAL_8]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_30:.*]] = hlfir.designate %[[VAL_29]]{"u2t3"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_31:.*]] = hlfir.designate %[[VAL_30]]{"t2"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_32:.*]] = hlfir.designate %[[VAL_31]]{"t1"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_33:.*]] = hlfir.designate %[[VAL_32]]{"t1i"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt1
! CHECK:           %[[VAL_34:.*]]:2 = hlfir.copy_in %[[VAL_33]] to %[[VAL_3]] : (!fir.box<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<5xi32>>>>) -> (!fir.box<!fir.array<5xi32>>, i1)
! CHECK:           %[[VAL_35:.*]] = fir.box_addr %[[VAL_34]]#0 : (!fir.box<!fir.array<5xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:           fir.call @_QPfoo5(%[[VAL_35]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()

  call foo6(u3va%u2%u2t3%t2%t1%t1i) ! different syntax for the previous value
! CHECK:           %[[VAL_36:.*]] = hlfir.designate %[[VAL_9]]#0{"u2"}   shape %[[VAL_8]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_37:.*]] = hlfir.designate %[[VAL_36]]{"u2t3"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_38:.*]] = hlfir.designate %[[VAL_37]]{"t2"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_39:.*]] = hlfir.designate %[[VAL_38]]{"t1"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_40:.*]] = hlfir.designate %[[VAL_39]]{"t1i"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt1
! CHECK:           %[[VAL_41:.*]]:2 = hlfir.copy_in %[[VAL_40]] to %[[VAL_2]] : (!fir.box<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<5xi32>>>>) -> (!fir.box<!fir.array<5xi32>>, i1)
! CHECK:           %[[VAL_42:.*]] = fir.box_addr %[[VAL_41]]#0 : (!fir.box<!fir.array<5xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:           fir.call @_QPfoo6(%[[VAL_42]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()

  call foo7(u3va%u2t4%t1i)
! CHECK:           %[[VAL_43:.*]] = hlfir.designate %[[VAL_9]]#0{"u2"}   shape %[[VAL_8]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_44:.*]] = hlfir.designate %[[VAL_43]]{"u2t4"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_45:.*]] = hlfir.designate %[[VAL_44]]{"t2"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_46:.*]] = hlfir.designate %[[VAL_45]]{"t1"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_47:.*]] = hlfir.designate %[[VAL_46]]{"t1i"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt1
! CHECK:           %[[VAL_48:.*]]:2 = hlfir.copy_in %[[VAL_47]] to %[[VAL_1]] : (!fir.box<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<5xi32>>>>) -> (!fir.box<!fir.array<5xi32>>, i1)
! CHECK:           %[[VAL_49:.*]] = fir.box_addr %[[VAL_48]]#0 : (!fir.box<!fir.array<5xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:           fir.call @_QPfoo7(%[[VAL_49]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()

  call foo8(u3va%u2t4%t2i)
! CHECK:           %[[VAL_50:.*]] = hlfir.designate %[[VAL_9]]#0{"u2"}   shape %[[VAL_8]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3
! CHECK:           %[[VAL_51:.*]] = hlfir.designate %[[VAL_50]]{"u2t4"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTu2
! CHECK:           %[[VAL_52:.*]] = hlfir.designate %[[VAL_51]]{"t2"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt3
! CHECK:           %[[VAL_53:.*]] = hlfir.designate %[[VAL_52]]{"t2i"}   shape %[[VAL_8]] : (!fir.box<!fir.array<5x!fir.type<_QFextended_type_componentsTt2
! CHECK:           %[[VAL_54:.*]]:2 = hlfir.copy_in %[[VAL_53]] to %[[VAL_0]] : (!fir.box<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<5xi32>>>>) -> (!fir.box<!fir.array<5xi32>>, i1)
! CHECK:           %[[VAL_55:.*]] = fir.box_addr %[[VAL_54]]#0 : (!fir.box<!fir.array<5xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:           fir.call @_QPfoo8(%[[VAL_55]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()
end subroutine extended_type_components
