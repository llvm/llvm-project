! Test lowering of structure constructors of derived types with allocatable component
! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefixes=HLFIR %s

module m_struct_ctor
  implicit none
  type t_alloc
    real :: x
    integer, allocatable :: a(:)
  end type

contains
  subroutine test_alloc1(y)
    real :: y
    call print_alloc_comp(t_alloc(x=y, a=null()))
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc1(
! HLFIR-SAME:      %[[ARG_0:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}) {
! HLFIR:    %[[VAL_0:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! HLFIR:    %[[VAL_1:.*]] = fir.address_of(@_QMm_struct_ctorE.n.x) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_1:.*]] = arith.constant 1 : index
! HLFIR:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}}"_QMm_struct_ctorE.n.x"
! HLFIR:    %[[VAL_3:.*]] = fir.address_of(@_QMm_struct_ctorE.n.a) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_2:.* ]]= arith.constant 1 : index
! HLFIR:    %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {{.*}}"_QMm_struct_ctorE.n.a"
! HLFIR:    %[[VAL_5:.*]] = fir.address_of(@_QMm_struct_ctorE.n.t_alloc) : !fir.ref<!fir.char<1,7>>
! HLFIR:    %[[CONS_3:.*]] = arith.constant 7 : index
! HLFIR:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {{.*}}"_QMm_struct_ctorE.n.t_alloc"
! HLFIR:    %[[VAL_7:.*]] = fir.address_of(@_QMm_struct_ctorE.c.t_alloc)
! HLFIR:    %[[CONS_4:.*]] = arith.constant 0 : index
! HLFIR:    %[[CONS_5:.*]] = arith.constant 2 : index
! HLFIR:    %[[VAL_8:.*]] = fir.shape_shift %[[CONS_4]], %[[CONS_5]] : (index, index) -> !fir.shapeshift<1>
! HLFIR:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {{.*}}"_QMm_struct_ctorE.c.t_alloc"
! HLFIR:    %[[VAL_10:.*]] = fir.address_of(@_QMm_struct_ctorE.dt.t_alloc)
! HLFIR:    %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {{.*}}"_QMm_struct_ctorE.dt.t_alloc"
! HLFIR:    %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG_0]] {uniq_name = "_QMm_struct_ctorFtest_alloc1Ey"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR:    %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! HLFIR:    %[[VAL_14:.*]] = fir.embox %[[VAL_13]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    %[[VAL_15:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! HLFIR:    %[[CONS_6:.*]] = arith.constant {{.*}} : i32
! HLFIR:    %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
! HLFIR:    %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! HLFIR:    %{{.*}} = fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[CONS_6]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
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
! HLFIR:    %[[VAL_1:.*]] = fir.address_of(@_QMm_struct_ctorE.n.x) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_1:.*]] = arith.constant 1 : index
! HLFIR:    %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[CONS_1]] {{.*}}"_QMm_struct_ctorE.n.x"
! HLFIR:    %[[VAL_3:.*]] = fir.address_of(@_QMm_struct_ctorE.n.a) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_2:.*]] = arith.constant 1 : index
! HLFIR:    %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[CONS_2]] {{.*}}"_QMm_struct_ctorE.n.a"
! HLFIR:    %[[VAL_5:.*]] = fir.address_of(@_QMm_struct_ctorE.n.t_alloc) : !fir.ref<!fir.char<1,7>>
! HLFIR:    %[[CONS_3:.*]] = arith.constant 7 : index
! HLFIR:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[CONS_3]] {{.*}}"_QMm_struct_ctorE.n.t_alloc"
! HLFIR:    %[[VAL_7:.*]] = fir.address_of(@_QMm_struct_ctorE.c.t_alloc)
! HLFIR:    %[[CONS_4:.*]] = arith.constant 0 : index
! HLFIR:    %[[CONS_5:.*]] = arith.constant 2 : index
! HLFIR:    %[[VAL_8:.*]] = fir.shape_shift %[[CONS_4]], %[[CONS_5]] : (index, index) -> !fir.shapeshift<1>
! HLFIR:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {{.*}}"_QMm_struct_ctorE.c.t_alloc"
! HLFIR:    %[[VAL_10:.*]] = fir.address_of(@_QMm_struct_ctorE.dt.t_alloc)
! HLFIR:    %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {{.*}}"_QMm_struct_ctorE.dt.t_alloc"
! HLFIR:    %[[CONS_6:.*]] = arith.constant 5 : index
! HLFIR:    %[[VAL_12:.*]] = fir.shape %[[CONS_6]] : (index) -> !fir.shape<1>
! HLFIR:    %[[VAL_13:.*]]:2 = hlfir.declare %[[ARG_1]](%[[VAL_12]]) {uniq_name = "_QMm_struct_ctorFtest_alloc2Eb"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! HLFIR:    %[[VAL_14:.*]]:2 = hlfir.declare %[[ARG_0]] {uniq_name = "_QMm_struct_ctorFtest_alloc2Ey"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR:    %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! HLFIR:    %[[VAL_16:.*]] = fir.embox %[[VAL_15]]#0 : (!fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    %[[VAL_17:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! HLFIR:    %[[CONS_7:.*]] = arith.constant {{.*}} : i32
! HLFIR:    %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
! HLFIR:    %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! HLFIR:    {{.*}} = fir.call @_FortranAInitialize(%[[VAL_18]], %[[VAL_19]], %[[CONS_7]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
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
! HLFIR:    %[[VAL_0:.*]] = fir.address_of(@_QMm_struct_ctorE.n.x) : !fir.ref<!fir.char<1>>
! HLFIR:    %c1 = arith.constant 1 : index
! HLFIR:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %c1 {{.*}}"_QMm_struct_ctorE.n.x"
! HLFIR:    %[[VAL_2:.*]] = fir.address_of(@_QMm_struct_ctorE.n.a) : !fir.ref<!fir.char<1>>
! HLFIR:    %c1_0 = arith.constant 1 : index
! HLFIR:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %c1_0 {{.*}}"_QMm_struct_ctorE.n.a"
! HLFIR:    %[[VAL_4:.*]] = fir.address_of(@_QMm_struct_ctorE.n.t_alloc) : !fir.ref<!fir.char<1,7>>
! HLFIR:    %c7 = arith.constant 7 : index
! HLFIR:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %c7 {{.*}}"_QMm_struct_ctorE.n.t_alloc"
! HLFIR:    %[[VAL_6:.*]] = fir.address_of(@_QMm_struct_ctorE.c.t_alloc)
! HLFIR:    %c0 = arith.constant 0 : index
! HLFIR:    %c2 = arith.constant 2 : index
! HLFIR:    %[[VAL_7:.*]] = fir.shape_shift %c0, %c2 : (index, index) -> !fir.shapeshift<1>
! HLFIR:    %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_7]]) {{.*}}"_QMm_struct_ctorE.c.t_alloc"
! HLFIR:    %[[VAL_9:.*]] = fir.address_of(@_QMm_struct_ctorE.dt.t_alloc)
! HLFIR:    %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {{.*}}"_QMm_struct_ctorE.dt.t_alloc"
! HLFIR:    %[[VAL_11:.*]] = fir.address_of(@_QMm_struct_ctorFtest_alloc3Et1) : !fir.ref<!fir.type<_QMm_struct_ctorTt_alloc{x:f32,a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! HLFIR:    {{.*}}:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QMm_struct_ctorFtest_alloc3Et1"}
! HLFIR:    return
! HLFIR:  }
  end subroutine

  subroutine test_alloc4()
    integer, pointer :: p(:)
    type(t_alloc) :: t1 = t_alloc(x=5, a=null(p))
! HLFIR-LABEL:  func.func @_QMm_struct_ctorPtest_alloc4() {
! HLFIR:    %[[VAL_0:.*]] = fir.address_of(@_QMm_struct_ctorE.n.x) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_1:.*]] = arith.constant 1 : index
! HLFIR:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[CONS_1]] {{.*}}"_QMm_struct_ctorE.n.x"
! HLFIR:    %[[VAL_2:.*]] = fir.address_of(@_QMm_struct_ctorE.n.a) : !fir.ref<!fir.char<1>>
! HLFIR:    %[[CONS_2:.*]] = arith.constant 1 : index
! HLFIR:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[CONS_2]] {{.*}}"_QMm_struct_ctorE.n.a"
! HLFIR:    %[[VAL_4:.*]] = fir.address_of(@_QMm_struct_ctorE.n.t_alloc) : !fir.ref<!fir.char<1,7>>
! HLFIR:    %[[CONS_3:.*]] = arith.constant 7 : index
! HLFIR:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[CONS_3]] {{.*}}"_QMm_struct_ctorE.n.t_alloc"
! HLFIR:    %[[VAL_6:.*]] = fir.address_of(@_QMm_struct_ctorE.c.t_alloc)
! HLFIR:    %[[CONS_4:.*]] = arith.constant 0 : index
! HLFIR:    %[[CONS_5:.*]] = arith.constant 2 : index
! HLFIR:    %[[VAL_7:.*]] = fir.shape_shift %[[CONS_4]], %[[CONS_5]] : (index, index) -> !fir.shapeshift<1>
! HLFIR:    %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_7]]) {{.*}}"_QMm_struct_ctorE.c.t_alloc"
! HLFIR:    %[[VAL_9:.*]] = fir.address_of(@_QMm_struct_ctorE.dt.t_alloc)
! HLFIR:    %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {{.*}}"_QMm_struct_ctorE.dt.t_alloc"
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
