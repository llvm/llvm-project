! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests various aspect of the lowering of polymorphic entities.

module polymorphic_test
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: print
  end type

  type, extends(p1) :: p2
    real :: c
  end type

  type r1
    real, pointer :: rp(:) => null()
  end type

  type c1
    character(2) :: tmp = 'c1'
  contains
    procedure :: get_tmp
  end type

  contains

  ! Test correct access to polymorphic entity component.
  subroutine component_access(p)
    class(p1) :: p
    print*, p%a
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcomponent_access(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "p"}) {
! CHECK: %[[FIELD:.*]] = fir.field_index a, !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[FIELD]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<i32>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[LOAD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1

  subroutine print(this)
    class(p1) :: this
  end subroutine

  ! Test embox of fir.type to fir.class to be passed-object.
  subroutine check()
    type(p1) :: t1
    type(p2) :: t2
    call t1%print()
    call t2%print()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcheck()
! CHECK: %[[DT1:.*]] = fir.alloca !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}> {bindc_name = "t1", uniq_name = "_QMpolymorphic_testFcheckEt1"}
! CHECK: %[[DT2:.*]] = fir.alloca !fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}> {bindc_name = "t2", uniq_name = "_QMpolymorphic_testFcheckEt2"}
! CHECK: %[[CLASS1:.*]] = fir.embox %[[DT1]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMpolymorphic_testPprint(%[[CLASS1]]) {{.*}}: (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: %[[BOX2:.*]] = fir.embox %[[DT2]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>
! CHECK: %[[CLASS2:.*]] = fir.convert %[[BOX2]] : (!fir.class<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> 
! CHECK: fir.call @_QMpolymorphic_testPprint(%[[CLASS2]]) {{.*}}: (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()

  subroutine test_allocate_unlimited_polymorphic_non_derived()
    class(*), pointer :: u
    allocate(integer::u)
  end subroutine

! CHECK-LABEL: test_allocate_unlimited_polymorphic_non_derived
! CHECK-NOT: _FortranAPointerNullifyDerived
! CHECK: fir.call @_FortranAPointerAllocate

  function test_fct_ret_class()
    class(p1), pointer :: test_fct_ret_class
  end function

  subroutine call_fct()
    class(p1), pointer :: p
    p => test_fct_ret_class()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_fct_ret_class() -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: return %{{.*}} : !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

! CHECK-lABEL: func.func @_QMpolymorphic_testPcall_fct()
! CHECK: %[[RESULT:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {bindc_name = ".result"}
! CHECK: %[[CALL_RES:.*]] = fir.call @_QMpolymorphic_testPtest_fct_ret_class() {{.*}}: () -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: fir.save_result %[[CALL_RES]] to %[[RESULT]] : !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>

  subroutine implicit_loop_with_polymorphic()
    class(p1), allocatable :: p(:)
    allocate(p(3))
    p%a = [ 1, 2, 3 ]
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPimplicit_loop_with_polymorphic() {
! CHECK: %{{.*}} = fir.array_load %{{.*}}(%{{.*}}) [%{{.*}}] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK: %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<?xi32>) {
! CHECK: %{{.*}} = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<3xi32>, index) -> i32
! CHECK: %{{.*}} = fir.array_update %{{.*}}, %{{.*}}, %{{.*}} : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK: }
! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %{{.*}}[%{{.*}}] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, !fir.slice<1>

  subroutine polymorphic_to_nonpolymorphic(p)
    class(p1), pointer :: p(:)
    type(p1), allocatable, target :: t(:)
    t = p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPpolymorphic_to_nonpolymorphic
! Just checking that FIR is generated without error.

! Test that lowering does not crash for function return with unlimited
! polymoprhic value.

  function up_ret()
    class(*), pointer :: up_ret(:)
  end function

! CHECK-LABEL: func.func @_QMpolymorphic_testPup_ret() -> !fir.class<!fir.ptr<!fir.array<?xnone>>> {

  subroutine call_up_ret()
    class(*), pointer :: p(:)
    p => up_ret()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_up_ret() {
! CHECK:         %{{.*}} = fir.call @_QMpolymorphic_testPup_ret() {{.*}} : () -> !fir.class<!fir.ptr<!fir.array<?xnone>>>

  subroutine rebox_f32_to_none(r)
    class(r1) :: r
    class(*), pointer :: p(:)
    p => r%rp
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPrebox_f32_to_none(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "r"}) {
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFrebox_f32_to_noneEp"}
! CHECK: %[[FIELD_RP:.*]] = fir.field_index rp, !fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK: %[[COORD_RP:.*]] = fir.coordinate_of %[[ARG0]], %[[FIELD_RP]] : (!fir.class<!fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[LOADED_RP:.*]] = fir.load %[[COORD_RP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[RP_DIMS:.*]]:3 = fir.box_dims %[[LOADED_RP]], %[[C0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[SHIFT:.*]] = fir.shift %[[RP_DIMS]]#0 : (index) -> !fir.shift<1>
! CHECK: %[[REBOX_TO_BOX:.*]] = fir.rebox %[[LOADED_RP]](%[[SHIFT]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK: %[[REBOX_TO_UP:.*]] = fir.rebox %[[REBOX_TO_BOX]] : (!fir.box<!fir.array<?xf32>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK: fir.store %[[REBOX_TO_UP]] to %[[P]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK: return

! Test that the fir.dispatch operation is created with the correct pass object
! and the pass_arg_pos attribute is incremented correctly when character
! function result is added as argument.

  function get_tmp(this)
    class(c1) :: this
    character(2) :: get_tmp
    get_tmp = this%tmp
  end function

  subroutine call_get_tmp(c)
    class(c1) :: c
    print*, c%get_tmp()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_get_tmp(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTc1{tmp:!fir.char<1,2>}>> {fir.bindc_name = "c"}) {
! CHECK: %{{.*}} = fir.dispatch "get_tmp"(%[[ARG0]] : !fir.class<!fir.type<_QMpolymorphic_testTc1{tmp:!fir.char<1,2>}>>) (%{{.*}}, %{{.*}}, %[[ARG0]] : !fir.ref<!fir.char<1,2>>, index, !fir.class<!fir.type<_QMpolymorphic_testTc1{tmp:!fir.char<1,2>}>>) -> !fir.boxchar<1> {pass_arg_pos = 2 : i32}

end module
