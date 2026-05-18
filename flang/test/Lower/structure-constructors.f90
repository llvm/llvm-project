! Test lowering of structure constructors
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m_struct_ctor
  implicit none
  type t_simple
    real :: x
  end type
  type t_char_scalar
    real :: x
    character(3) :: c
  end type
  type t_array
    real :: x
    integer :: i(5)
  end type
  type t_char_array
    real :: x
    character(3) :: c(5)
  end type
  type t_ptr
    real :: x
    integer, pointer :: p(:,:)
  end type
  type t_nested
    real :: x
    type(t_array) :: dt
  end type
contains
  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple(
  ! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32>{{.*}})
  subroutine test_simple(x)
    real :: x
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_simple{x:f32}>
    ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_simpleEx"}
    ! CHECK: %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
    ! CHECK: fir.call @_FortranAInitialize(
    ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
    ! CHECK: %[[val:.*]] = fir.load %[[xdecl]]#0 : !fir.ref<f32>
    ! CHECK: hlfir.assign %[[val]] to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
    call print_simple(t_simple(x=x))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_scalar(
  ! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32>{{.*}})
  subroutine test_char_scalar(x)
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_char_scalarEx"}
    ! CHECK: %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
    ! CHECK: fir.call @_FortranAInitialize(
    ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
    ! CHECK: %[[val:.*]] = fir.load %[[xdecl]]#0 : !fir.ref<f32>
    ! CHECK: hlfir.assign %[[val]] to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
    ! CHECK: %[[ccoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"c"}{{.*}} typeparams %{{.*}} : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>>, index) -> !fir.ref<!fir.char<1,3>>
    ! CHECK: %[[cst:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,3>>
    ! CHECK: %[[cstdecl:.*]]:2 = hlfir.declare %[[cst]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQ{{.*}}"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
    ! CHECK: hlfir.assign %[[cstdecl]]#0 to %[[ccoor]] temporary_lhs : !fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>
    real :: x
    call print_char_scalar(t_char_scalar(x=x, c="abc"))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple_array(
  ! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.array<5xi32>>{{.*}})
  subroutine test_simple_array(x, j)
    real :: x
    integer :: j(5)
    call print_simple_array(t_array(x=x, i=2*j))
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[shape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
    ! CHECK: %[[jdecl:.*]]:2 = hlfir.declare %[[arg1]](%[[shape]]){{.*}}{uniq_name = "_QMm_struct_ctorFtest_simple_arrayEj"}
    ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_simple_arrayEx"}
    ! CHECK: %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
    ! CHECK: fir.call @_FortranAInitialize(
    ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
    ! CHECK: %[[xval:.*]] = fir.load %[[xdecl]]#0 : !fir.ref<f32>
    ! CHECK: hlfir.assign %[[xval]] to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
    ! CHECK: %[[icoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"i"}
    ! CHECK: %[[c2:.*]] = arith.constant 2 : i32
    ! CHECK: %[[elem:.*]] = hlfir.elemental %[[shape]] unordered : (!fir.shape<1>) -> !hlfir.expr<5xi32> {
    ! CHECK:   %[[idx:.*]] = hlfir.designate %[[jdecl]]#0 (%{{.*}})  : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
    ! CHECK:   %[[jval:.*]] = fir.load %[[idx]] : !fir.ref<i32>
    ! CHECK:   %[[mul:.*]] = arith.muli %[[c2]], %[[jval]] : i32
    ! CHECK:   hlfir.yield_element %[[mul]] : i32
    ! CHECK: }
    ! CHECK: hlfir.assign %[[elem]] to %[[icoor]] temporary_lhs : !hlfir.expr<5xi32>, !fir.ref<!fir.array<5xi32>>
    ! CHECK: hlfir.destroy %[[elem]] : !hlfir.expr<5xi32>
  end subroutine

! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_array(
! CHECK-SAME:  %[[arg0:.*]]: !fir.ref<f32>{{.*}}, %[[arg1:.*]]: !fir.boxchar<1>{{.*}}) {
  subroutine test_char_array(x, c1)
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[arg1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[c1ref:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<5x!fir.char<1,3>>>
  ! CHECK: %[[c1decl:.*]]:2 = hlfir.declare %[[c1ref]](%{{.*}}) typeparams %{{.*}}{{.*}}{uniq_name = "_QMm_struct_ctorFtest_char_arrayEc1"}
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_char_arrayEx"}
  ! CHECK: %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
  ! CHECK: fir.call @_FortranAInitialize(
  ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
  ! CHECK: hlfir.assign %{{.*}} to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
  ! CHECK: %[[ccoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"c"}
  ! CHECK: %[[elem:.*]] = hlfir.elemental %{{.*}} typeparams %{{.*}} unordered : (!fir.shape<1>, i64) -> !hlfir.expr<5x!fir.char<1,?>> {
  ! CHECK:   %[[carg:.*]] = hlfir.designate %[[c1decl]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.ref<!fir.array<5x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:   %[[setlen:.*]] = hlfir.set_length %[[carg]] len %{{.*}} : (!fir.ref<!fir.char<1,3>>, i64) -> !hlfir.expr<!fir.char<1,3>>
  ! CHECK:   hlfir.yield_element %[[setlen]] : !hlfir.expr<!fir.char<1,3>>
  ! CHECK: }
  ! CHECK: hlfir.assign %[[elem]] to %[[ccoor]] temporary_lhs : !hlfir.expr<5x!fir.char<1,?>>, !fir.ref<!fir.array<5x!fir.char<1,3>>>
  ! CHECK: fir.call @_QMm_struct_ctorPprint_char_array(%[[tmpdecl]]#0)

    real :: x
    character(3) :: c1(5)
    call print_char_array(t_char_array(x=x, c=c1))
    ! CHECK: return
    ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_ptr(
  ! CHECK-SAME:    %[[arg0:.*]]: !fir.ref<f32>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x?xi32>> {{{.*}}, fir.target}) {
  ! CHECK:         %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
  ! CHECK:         %[[adecl:.*]]:2 = hlfir.declare %[[arg1]]{{.*}}{fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMm_struct_ctorFtest_ptrEa"}
  ! CHECK:         %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_ptrEx"}
  ! CHECK:         %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
  ! CHECK:         fir.call @_FortranAInitialize(
  ! CHECK:         %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
  ! CHECK:         hlfir.assign %{{.*}} to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
  ! CHECK:         %[[pcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<pointer>}
  ! CHECK:         %[[slice:.*]] = hlfir.designate %[[adecl]]#0 (%{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?x?xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<2x3xi32>>
  ! CHECK:         %[[rebox:.*]] = fir.rebox %[[slice]] : (!fir.box<!fir.array<2x3xi32>>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
  ! CHECK:         fir.store %[[rebox]] to %[[pcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
  ! CHECK:         fir.call @_QMm_struct_ctorPprint_ptr(%[[tmpdecl]]#0)
  ! CHECK:         return
  ! CHECK:       }

  subroutine test_ptr(x, a)
    real :: x
    integer, target :: a(:, :)
    call print_ptr(t_ptr(x=x, p=a(1:4:2, 1:3:1)))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_nested(
  ! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
  subroutine test_nested(x, d)
    real :: x
    type(t_array) :: d
  ! CHECK:         %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>
  ! CHECK:         %[[ddecl:.*]]:2 = hlfir.declare %[[arg1]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_nestedEd"}
  ! CHECK:         %[[xdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMm_struct_ctorFtest_nestedEx"}
  ! CHECK:         %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
  ! CHECK:         fir.call @_FortranAInitialize(
  ! CHECK:         %[[xcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"x"}
  ! CHECK:         %[[xval:.*]] = fir.load %[[xdecl]]#0 : !fir.ref<f32>
  ! CHECK:         hlfir.assign %[[xval]] to %[[xcoor]] temporary_lhs : f32, !fir.ref<f32>
  ! CHECK:         %[[dtcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"dt"}
  ! CHECK:         hlfir.assign %[[ddecl]]#0 to %[[dtcoor]] temporary_lhs : !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
    call print_nested(t_nested(x=x, dt=d))
  end subroutine

  subroutine print_simple(t)
    type(t_simple) :: t
    print *, t%x
  end subroutine
  subroutine print_char_scalar(t)
    type(t_char_scalar) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_simple_array(t)
    type(t_array) :: t
    print *, t%x, t%i
  end subroutine
  subroutine print_char_array(t)
    type(t_char_array) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_ptr(t)
    type(t_ptr) :: t
    print *, t%x, t%p
  end subroutine
  subroutine print_nested(t)
    type(t_nested) :: t
    print *, t%x, t%dt%x, t%dt%i
  end subroutine

end module

  use m_struct_ctor
  integer, target :: i(4,3) = reshape([1,2,3,4,5,6,7,8,9,10,11,12], [4,3])
  call test_simple(42.)
  call test_char_scalar(42.)
  call test_simple_array(42., [1,2,3,4,5])
  call test_char_array(42., ["abc", "def", "geh", "ijk", "lmn"])
  call test_ptr(42., i)
  call test_nested(42., t_array(x=43., i=[5,6,7,8,9]))
end

! CHECK-LABEL: func.func @_QPtest_parent_component1() {
! CHECK:         %[[ro:.*]] = fir.address_of(@_QQro._QFtest_parent_component1Tmid.{{[0-9]+}}) : !fir.ref<!fir.type<_QFtest_parent_component1Tmid{base:!fir.type<_QFtest_parent_component1Tbase{x:i32,y:!fir.array<2xi32>}>,mask:!fir.logical<4>}>>
! CHECK:         %[[rodecl:.*]]:2 = hlfir.declare %[[ro]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QFtest_parent_component1Tmid.{{[0-9]+}}"}
! CHECK:         %[[expr:.*]] = hlfir.as_expr %[[rodecl]]#0 : (!fir.ref<!fir.type<_QFtest_parent_component1Tmid{base:!fir.type<_QFtest_parent_component1Tbase{x:i32,y:!fir.array<2xi32>}>,mask:!fir.logical<4>}>>) -> !hlfir.expr<!fir.type<_QFtest_parent_component1Tmid{base:!fir.type<_QFtest_parent_component1Tbase{x:i32,y:!fir.array<2xi32>}>,mask:!fir.logical<4>}>>
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[expr]] {adapt.valuebyref}
! CHECK:         fir.call @_QPprint_parent_component1(%[[assoc]]#0)
! CHECK:         hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2
! CHECK:         return
! CHECK:       }

subroutine test_parent_component1()
  type base
    integer :: x, y(2)
  end type base
  type, extends(base) :: mid
    logical :: mask
  end type mid

  call print_parent_component1(mid(base = base(1, [2, 3]), mask = .true.))
end

! CHECK-LABEL: func.func @_QPtest_parent_component2() {
! CHECK:         %[[tmp:.*]] = fir.alloca !fir.type<_QFtest_parent_component2Tmid{base:!fir.type<_QFtest_parent_component2Tbase{z:!fir.char<1,5>}>,mask:!fir.logical<4>}>
! CHECK:         %[[pv:.*]] = fir.address_of(@_QFtest_parent_component2Epv) : !fir.ref<!fir.type<_QFtest_parent_component2Tbase{z:!fir.char<1,5>}>>
! CHECK:         %[[pvdecl:.*]]:2 = hlfir.declare %[[pv]] {uniq_name = "_QFtest_parent_component2Epv"}
! CHECK:         %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
! CHECK:         fir.call @_FortranAInitialize(
! CHECK:         %[[basecoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"base"}
! CHECK:         hlfir.assign %[[pvdecl]]#0 to %[[basecoor]] temporary_lhs : !fir.ref<!fir.type<_QFtest_parent_component2Tbase{z:!fir.char<1,5>}>>, !fir.ref<!fir.type<_QFtest_parent_component2Tbase{z:!fir.char<1,5>}>>
! CHECK:         %[[maskcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"mask"}
! CHECK:         %[[t:.*]] = arith.constant true
! CHECK:         %[[lt:.*]] = fir.convert %[[t]] : (i1) -> !fir.logical<4>
! CHECK:         hlfir.assign %[[lt]] to %[[maskcoor]] temporary_lhs : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:         fir.call @_QPprint_parent_component2(%[[tmpdecl]]#0)
! CHECK:         return
! CHECK:       }

subroutine test_parent_component2()
  type base
    character(5) :: z
  end type base
  type, extends(base) :: mid
    logical :: mask
  end type mid
  type(base) :: pv = base("aaa")

  call print_parent_component2(mid(base = pv, mask = .true.))
end

! CHECK-LABEL: func.func @_QPtest_parent_component3(
! CHECK-SAME:                                       %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>>> {fir.bindc_name = "pp"}) {
! CHECK:         %[[tmp:.*]] = fir.alloca !fir.type<_QFtest_parent_component3Tmid{base:!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>,mask:!fir.logical<4>}>
! CHECK:         %[[ppdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_parent_component3Epp"}
! CHECK:         %[[tmpdecl:.*]]:2 = hlfir.declare %[[tmp]] {uniq_name = "ctor.temp"}
! CHECK:         fir.call @_FortranAInitialize(
! CHECK:         %[[basecoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"base"}
! CHECK:         %[[ppload:.*]] = fir.load %[[ppdecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>>>
! CHECK:         %[[ppaddr:.*]] = fir.box_addr %[[ppload]] : (!fir.box<!fir.ptr<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>>) -> !fir.ptr<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>
! CHECK:         hlfir.assign %[[ppaddr]] to %[[basecoor]] temporary_lhs : !fir.ptr<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>, !fir.ref<!fir.type<_QFtest_parent_component3Tbase{m:!fir.array<2x!fir.char<1,5>>}>>
! CHECK:         %[[maskcoor:.*]] = hlfir.designate %[[tmpdecl]]#0{"mask"}
! CHECK:         %[[t:.*]] = arith.constant true
! CHECK:         %[[lt:.*]] = fir.convert %[[t]] : (i1) -> !fir.logical<4>
! CHECK:         hlfir.assign %[[lt]] to %[[maskcoor]] temporary_lhs : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:         fir.call @_QPprint_parent_component3(%[[tmpdecl]]#0)
! CHECK:         return
! CHECK:       }

subroutine test_parent_component3(pp)
  type base
    character(5) :: m(2)
  end type base
  type, extends(base) :: mid
    logical :: mask
  end type mid
  type(base), pointer :: pp

  call print_parent_component3(mid(base = pp, mask = .true.))
end
