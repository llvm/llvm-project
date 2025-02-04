! Test lowering of structure constructors of derived types with allocatable component
! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefixes=HLFIR %s

module m_struct_ctor
  implicit none

  type t_alloc
    real :: x
    integer, allocatable :: a(:)
  end type

  type t_alloc_char
    character(:), allocatable :: a
  end type

  type t_alloc_char_cst_len
    character(2), allocatable :: a
  end type

contains
  subroutine test_alloc1(y)
    real :: y
    call print_alloc_comp(t_alloc(x=y, a=null()))
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc1(
! HLFIR-SAME:      %[[ARG_0:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}) {
! HLFIR:    %[[VAL_0:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! HLFIR:    %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMm_struct_ctorFtest_alloc1Ey"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR:    %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! HLFIR:    %[[VAL_14:.*]] = fir.embox %[[VAL_13]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    %[[VAL_15:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! HLFIR:    %[[CONS_6:.*]] = arith.constant {{.*}} : i32
! HLFIR:    %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
! HLFIR:    %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! HLFIR:    fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[CONS_6]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
! HLFIR:    %[[VAL_18:.*]] = hlfir.designate %[[VAL_13]]#0{"x"}   : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<f32>
! HLFIR:    %[[VAL_19:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<f32>
! HLFIR:    hlfir.assign %[[VAL_19]] to %[[VAL_18]] temporary_lhs : f32, !fir.ref<f32>
! HLFIR:    fir.call @_QPprint_alloc_comp(%[[VAL_13]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> ()
! HLFIR:    return
! HLFIR:  }
  end subroutine

  subroutine test_alloc2(y, b)
    real :: y
    integer :: b(5)
    call print_alloc_comp(t_alloc(x=y, a=b))
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc2
! HLFIR-SAME:     (%[[ARG_0:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}, %[[ARG_1:.*]]: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "b"}) {
! HLFIR:    %[[VAL_0:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! HLFIR:    %[[CONS_6:.*]] = arith.constant 5 : index
! HLFIR:    %[[VAL_12:.*]] = fir.shape %[[CONS_6]] : (index) -> !fir.shape<1>
! HLFIR:    %[[VAL_13:.*]]:2 = hlfir.declare %[[ARG_1]](%[[VAL_12]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QMm_struct_ctorFtest_alloc2Eb"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! HLFIR:    %[[VAL_14:.*]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMm_struct_ctorFtest_alloc2Ey"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR:    %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! HLFIR:    %[[VAL_16:.*]] = fir.embox %[[VAL_15]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    %[[VAL_17:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! HLFIR:    %[[CONS_7:.*]] = arith.constant {{.*}} : i32
! HLFIR:    %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
! HLFIR:    %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! HLFIR:    fir.call @_FortranAInitialize(%[[VAL_18]], %[[VAL_19]], %[[CONS_7]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
! HLFIR:    %[[VAL_20:.*]] = hlfir.designate %[[VAL_15]]#0{"x"}   : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<f32>
! HLFIR:    %[[VAL_21:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<f32>
! HLFIR:    hlfir.assign %[[VAL_21]] to %[[VAL_20]] temporary_lhs : f32, !fir.ref<f32>
! HLFIR:    %[[VAL_22:.*]] = hlfir.designate %[[VAL_15]]#0{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:    hlfir.assign %[[VAL_13]]#0 to %[[VAL_22]] realloc temporary_lhs : !fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:    fir.call @_QPprint_alloc_comp(%[[VAL_15]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> ()
! HLFIR:    return
! HLFIR:  }
  end subroutine

  subroutine test_alloc3()
    type(t_alloc) :: t1 = t_alloc(x=5, a=null())
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc3() {
! HLFIR:    %[[VAL_11:.*]] = fir.address_of(@_QMm_struct_ctorFtest_alloc3Et1) : !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    {{.*}}:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QMm_struct_ctorFtest_alloc3Et1"}
! HLFIR:    return
! HLFIR:  }
  end subroutine

  subroutine test_alloc4()
    integer, pointer :: p(:)
    type(t_alloc) :: t1 = t_alloc(x=5, a=null(p))
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc4() {
! HLFIR:    %[[VAL_11:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p", uniq_name = "_QMm_struct_ctorFtest_alloc4Ep"}
! HLFIR:    %[[VAL_12:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! HLFIR:    %[[CONS_6:.*]] = arith.constant 0 : index
! HLFIR:    %[[VAL_13:.*]] = fir.shape %[[CONS_6]] : (index) -> !fir.shape<1>
! HLFIR:    %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! HLFIR:    fir.store %[[VAL_14]] to %[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR:    %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_11]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm_struct_ctorFtest_alloc4Ep"}
! HLFIR:    %[[VAL_16:.*]] = fir.address_of(@_QMm_struct_ctorFtest_alloc4Et1) : !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QMm_struct_ctorFtest_alloc4Et1"}
! HLFIR:    return
! HLFIR:  }
  end subroutine

end module m_struct_ctor

subroutine test_character_1()
  use m_struct_ctor, only : t_alloc_char
  interface
    subroutine takes_ta_alloc_char(x)
      import t_alloc_char
      type(t_alloc_char) :: x
    end subroutine
  end interface
  call takes_ta_alloc_char(t_alloc_char("hello"))
end subroutine
! HLFIR-LABEL:   func.func @_QPtest_character_1() {
! HLFIR:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>
! HLFIR:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>)
! HLFIR:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>
! HLFIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>) -> !fir.box<none>
! HLFIR:           fir.call @_FortranAInitialize(%[[VAL_5]],
! HLFIR:           %[[VAL_8:.*]] = hlfir.designate %[[VAL_1]]#0{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! HLFIR:           %[[VAL_9:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! HLFIR:           %[[VAL_10:.*]] = arith.constant 5 : index
! HLFIR:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! HLFIR:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_8]] realloc temporary_lhs : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! HLFIR:           fir.call @_QPtakes_ta_alloc_char(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char{a:!fir.box<!fir.heap<!fir.char<1,?>>>}>>) -> ()

subroutine test_character_2()
  use m_struct_ctor, only : t_alloc_char_cst_len
  interface
    subroutine takes_ta_alloc_char_cst_len(x)
      import t_alloc_char_cst_len
      type(t_alloc_char_cst_len) :: x
    end subroutine
  end interface
  call takes_ta_alloc_char_cst_len(t_alloc_char_cst_len("hello"))
end subroutine
! HLFIR-LABEL:   func.func @_QPtest_character_2() {
! HLFIR:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>
! HLFIR:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>)
! HLFIR:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>
! HLFIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>) -> !fir.box<none>
! HLFIR:           fir.call @_FortranAInitialize(%[[VAL_5]],
! HLFIR:           %[[VAL_8:.*]] = arith.constant 2 : index
! HLFIR:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_1]]#0{"a"}   typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,2>>>>
! HLFIR:           %[[VAL_10:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! HLFIR:           %[[VAL_11:.*]] = arith.constant 5 : index
! HLFIR:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]] typeparams %[[VAL_11]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! HLFIR:           hlfir.assign %[[VAL_12]]#0 to %[[VAL_9]] realloc keep_lhs_len temporary_lhs : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,2>>>>
! HLFIR:           fir.call @_QPtakes_ta_alloc_char_cst_len(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc_char_cst_len{a:!fir.box<!fir.heap<!fir.char<1,2>>>}>>) -> ()
