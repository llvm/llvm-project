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

  type p3
    class(p3), pointer :: p(:)
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

  subroutine associate_up_pointer(r)
    class(r1) :: r
    class(*), pointer :: p(:)
    p => r%rp
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPassociate_up_pointer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "r"}) {
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFassociate_up_pointerEp"}
! CHECK: %[[FIELD_RP:.*]] = fir.field_index rp, !fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK: %[[COORD_RP:.*]] = fir.coordinate_of %[[ARG0]], %[[FIELD_RP]] : (!fir.class<!fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[LOAD_RP:.*]] = fir.load %[[COORD_RP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[REBOX_RP:.*]] = fir.rebox %[[LOAD_RP]](%{{.*}}) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK: %[[CONV_P:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[RP_BOX_NONE:.*]] = fir.convert %[[REBOX_RP]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociate(%[[CONV_P]], %[[RP_BOX_NONE]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
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

  subroutine sub_with_type_array(a)
    type(p1) :: a(:)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_type_array(%{{.*}}: !fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"})

  subroutine call_sub_with_type_array(p)
    class(p1), pointer :: p(:)
    call sub_with_type_array(p)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_sub_with_type_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"}) {
! CHECK: %[[CLASS:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[CLASS]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>) -> !fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: fir.call @_QMpolymorphic_testPsub_with_type_array(%[[REBOX]]) {{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> ()

  subroutine derived_type_assignment_with_class()
    type(p3) :: a
    type(p3), target :: b(10)
    a = p3(b)
  end subroutine

  subroutine takes_p1(p)
    class(p1), intent(in) :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtakes_p1

  subroutine no_reassoc_poly_value(a, i)
    class(p1), intent(in) :: a(:)
    integer :: i
    call takes_p1((a(i)))
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPno_reassoc_poly_value(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"}, %[[I:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
! CHECK:  %[[TEMP:.*]] = fir.alloca !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>
! CHECK:  %[[LOADED_I:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:  %[[I_I64:.*]] = fir.convert %[[LOADED_I]] : (i32) -> i64
! CHECK:  %[[C1:.*]] = arith.constant 1 : i64
! CHECK:  %[[IDX:.*]] = arith.subi %[[I_I64]], %[[C1]] : i64
! CHECK:  %[[COORD:.*]] = fir.coordinate_of %[[ARG0]], %[[IDX]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[TDESC:.*]] = fir.box_tdesc %[[ARG0]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.tdesc<none>
! CHECK:  %[[NO_REASSOC:.*]] = fir.no_reassoc %[[COORD]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[LOAD:.*]] = fir.load %[[NO_REASSOC]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  fir.store %[[LOAD]] to %[[TEMP]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[TEMP]] tdesc %[[TDESC]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.tdesc<none>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  fir.call @_QMpolymorphic_testPtakes_p1(%[[EMBOX]]) {{.*}} : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()

! Test pointer assignment with non polymorphic lhs and polymorphic rhs

  subroutine pointer_assign_parent(p)
    type(p2), target :: p
    type(p1), pointer :: tp
    tp => p%p1
  end subroutine

! First test is here to have a reference with non polymorphic on both sides.
! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_parent(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>> {fir.bindc_name = "p", fir.target}) {
! CHECK: %[[TP:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {bindc_name = "tp", uniq_name = "_QMpolymorphic_testFpointer_assign_parentEtp"}
! CHECK: %[[PTR:.*]] = fir.alloca !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {uniq_name = "_QMpolymorphic_testFpointer_assign_parentEtp.addr"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.store %[[ZERO]] to %[[PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[CONVERT:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>) -> !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.store %[[CONVERT]] to %[[PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

  subroutine pointer_assign_non_poly(p)
    class(p1), target :: p
    type(p1), pointer :: tp
    tp => p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_non_poly(
! CHECK-SAME: %arg0: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "p", fir.target}) {
! CHECK: %[[TP:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {bindc_name = "tp", uniq_name = "_QMpolymorphic_testFpointer_assign_non_polyEtp"}
! CHECK: %[[PTR:.*]] = fir.alloca !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {uniq_name = "_QMpolymorphic_testFpointer_assign_non_polyEtp.addr"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.store %[[ZERO]] to %[[PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: %[[CONVERT:.*]] = fir.convert %3 : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.store %[[CONVERT]] to %[[PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

  subroutine nullify_pointer_array(a)
    type(p3) :: a
    nullify(a%p)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPnullify_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>> {fir.bindc_name = "a"}) {
! CHECK: %[[FIELD_P:.*]] = fir.field_index p, !fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>
! CHECK: %[[COORD_P:.*]] = fir.coordinate_of %[[ARG0]], %[[FIELD_P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>>, !fir.field) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>>>>>
! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolymorphic_testE.dt.p3) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[CONV_P:.*]] = fir.convert %[[COORD_P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[CONV_TDESC:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[CONV_P]], %[[CONV_TDESC]], %[[C1]], %[[C0]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none

  subroutine up_input(a)
    class(*), intent(in) :: a
  end subroutine

  subroutine pass_trivial_to_up()
    call up_input('hello')
    call up_input(1)
    call up_input(2.5)
    call up_input(.true.)
    call up_input((-1.0,3))
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPpass_trivial_to_up() {
! CHECK: %[[CHAR:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,5>>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[CHAR]] : (!fir.ref<!fir.char<1,5>>) -> !fir.class<!fir.char<1,5>>
! CHECK: %[[CONVERT:.*]] = fir.convert %[[EMBOX]] : (!fir.class<!fir.char<1,5>>) -> !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPup_input(%[[CONVERT]]) {{.*}} : (!fir.class<none>) -> ()

! CHECK: %[[BOX_INT:.*]] = fir.embox %{{.*}} : (!fir.ref<i32>) -> !fir.class<i32>
! CHECK: %[[UP:.*]] = fir.convert %[[BOX_INT]] : (!fir.class<i32>) -> !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPup_input(%[[UP]]) {{.*}} : (!fir.class<none>) -> ()

! CHECK: %[[BOX_REAL:.*]] = fir.embox %{{.*}} : (!fir.ref<f32>) -> !fir.class<f32>
! CHECK: %[[UP:.*]] = fir.convert %[[BOX_REAL]] : (!fir.class<f32>) -> !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPup_input(%[[UP]]) {{.*}} : (!fir.class<none>) -> ()

! CHECK: %[[BOX_LOG:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.class<!fir.logical<4>>
! CHECK: %[[UP:.*]] = fir.convert %[[BOX_LOG]] : (!fir.class<!fir.logical<4>>) -> !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPup_input(%[[UP]]) {{.*}} : (!fir.class<none>) -> ()

! CHECK: %[[BOX_COMPLEX:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.complex<4>>) -> !fir.class<!fir.complex<4>>
! CHECK: %[[UP:.*]] = fir.convert %[[BOX_COMPLEX]] : (!fir.class<!fir.complex<4>>) -> !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPup_input(%[[UP]]) {{.*}} : (!fir.class<none>) -> ()

  subroutine assign_polymorphic_allocatable()
    type(p1), target :: t(10,20)
    class(p1), allocatable :: c(:,:)
    c = t
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPassign_polymorphic_allocatable() {
! CHECK: %[[C:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {bindc_name = "c", uniq_name = "_QMpolymorphic_testFassign_polymorphic_allocatableEc"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE_C:.*]] = fir.shape %[[C0]], %[[C0]] : (index, index) -> !fir.shape<2>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE_C]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>
! CHECK: fir.store %[[EMBOX]] to %[[C]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : index
! CHECK: %[[T:.*]] = fir.alloca !fir.array<10x20x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "t", fir.target, uniq_name = "_QMpolymorphic_testFassign_polymorphic_allocatableEt"}
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C10]], %[[C20]] : (index, index) -> !fir.shape<2>
! CHECK: %[[BOXED_T:.*]] = fir.embox %[[T]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x20x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x20x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[CONV_C:.*]] = fir.convert %[[C]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[CONV_BOXED_T:.*]] = fir.convert %[[BOXED_T]] : (!fir.box<!fir.array<10x20x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAssign(%[[CONV_C]], %[[CONV_BOXED_T]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK: return

end module
