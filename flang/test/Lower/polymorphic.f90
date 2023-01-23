! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests various aspect of the lowering of polymorphic entities.

module polymorphic_test
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: print
    procedure :: assign_p1_int
    procedure :: elemental_fct
    procedure :: elemental_sub
    procedure, pass(this) :: elemental_sub_pass
    procedure :: read_p1
    procedure :: write_p1
    generic :: read(formatted) => read_p1
    generic :: write(formatted) => write_p1
    generic :: assignment(=) => assign_p1_int
    procedure :: host_assoc
    procedure, pass(poly) :: lt
    generic :: operator(<) => lt
  end type

  type, extends(p1) :: p2
    real :: c = 10.5
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

  type outer
    type(p1) :: inner
  end type

  type non_extensible
    sequence
    integer :: d
  end type

  contains

  elemental subroutine assign_p1_int(lhs, rhs)
    class(p1), intent(inout) :: lhs
    integer, intent(in) :: rhs
    lhs%a = rhs
    lhs%b = rhs
  End Subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPhost_assoc(
! CHECK-SAME: %[[THIS:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {
! CHECK: %[[TUPLE:.*]] = fir.alloca tuple<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[POS_IN_TUPLE:.*]] = arith.constant 0 : i32
! CHECK: %[[COORD_OF_CLASS:.*]] = fir.coordinate_of %[[TUPLE]], %[[POS_IN_TUPLE]] : (!fir.ref<tuple<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, i32) -> !fir.ref<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[THIS]] to %[[COORD_OF_CLASS]] : !fir.ref<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: fir.call @_QMpolymorphic_testFhost_assocPinternal(%[[TUPLE]]) {{.*}} : (!fir.ref<tuple<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>) -> ()

  elemental integer function elemental_fct(this)
    class(p1), intent(in) :: this
    elemental_fct = this%a
  end function

  elemental subroutine elemental_sub(this)
    class(p1), intent(inout) :: this
    this%a = this%a * this%b
  end subroutine

  elemental subroutine elemental_sub_pass(c, this)
    integer, intent(in) :: c
    class(p1), intent(inout) :: this
    this%a = this%a * this%b + c
  end subroutine

  logical elemental function lt(i, poly)
    integer, intent(in) :: i
    class(p1), intent(in) :: poly
    lt = i < poly%a
  End Function

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
! CHECK:  %[[NO_REASSOC:.*]] = fir.no_reassoc %[[COORD]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[LOAD:.*]] = fir.load %[[NO_REASSOC]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  fir.store %[[LOAD]] to %[[TEMP]] : !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[TEMP]] source_box %[[ARG0]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
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

  subroutine pointer_assign_remap()
    class(p1), pointer :: a(:)
    class(p1), pointer :: p(:,:)
    class(p1), pointer :: q(:)
    allocate(a(100))
    p(1:10,1:10) => a
    q(0:99) => a
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_remap() {
! CHECK: %[[A:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {bindc_name = "a", uniq_name = "_QMpolymorphic_testFpointer_assign_remapEa"}
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFpointer_assign_remapEp"}
! CHECK: %[[Q:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {bindc_name = "q", uniq_name = "_QMpolymorphic_testFpointer_assign_remapEq"}
! CHECK: %[[C1_0:.*]] = arith.constant 1 : i64
! CHECK: %[[C10_0:.*]] = arith.constant 10 : i64
! CHECK: %[[C1_1:.*]] = arith.constant 1 : i64
! CHECK: %[[C10_1:.*]] = arith.constant 10 : i64
! CHECK: %[[LOAD_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[REBOX_A:.*]] = fir.rebox %[[LOAD_A]](%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[BOUND_ARRAY:.*]] = fir.alloca !fir.array<4xi64>
! CHECK: %[[ARRAY:.*]] = fir.undefined !fir.array<4xi64>
! CHECK: %[[ARRAY0:.*]] = fir.insert_value %[[ARRAY]], %[[C1_0]], [0 : index] : (!fir.array<4xi64>, i64) -> !fir.array<4xi64>
! CHECK: %[[ARRAY1:.*]] = fir.insert_value %[[ARRAY0]], %[[C10_0]], [1 : index] : (!fir.array<4xi64>, i64) -> !fir.array<4xi64>
! CHECK: %[[ARRAY2:.*]] = fir.insert_value %[[ARRAY1]], %[[C1_1]], [2 : index] : (!fir.array<4xi64>, i64) -> !fir.array<4xi64>
! CHECK: %[[ARRAY3:.*]] = fir.insert_value %[[ARRAY2]], %[[C10_1]], [3 : index] : (!fir.array<4xi64>, i64) -> !fir.array<4xi64>
! CHECK: fir.store %[[ARRAY3]] to %[[BOUND_ARRAY]] : !fir.ref<!fir.array<4xi64>>
! CHECK: %[[C4:.*]] = arith.constant 4 : index
! CHECK: %[[BOUND_ARRAY_SHAPE:.*]] = fir.shape %[[C4]] : (index) -> !fir.shape<1>
! CHECK: %[[BOXED_BOUND_ARRAY:.*]] = fir.embox %[[BOUND_ARRAY]](%[[BOUND_ARRAY_SHAPE]]) : (!fir.ref<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.box<!fir.array<4xi64>>
! CHECK: %[[ARG0:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[ARG1:.*]] = fir.convert %[[REBOX_A]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %[[ARG2:.*]] = fir.convert %[[BOXED_BOUND_ARRAY]] : (!fir.box<!fir.array<4xi64>>) -> !fir.box<none>
! CHECK:  %{{.*}} = fir.call @_FortranAPointerAssociateRemapping(%[[ARG0]], %[[ARG1]], %[[ARG2]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none

! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[C99:.*]] = arith.constant 99 : i64
! CHECK: %[[LOAD_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>
! CHECK: %[[REBOX_A:.*]] = fir.rebox %[[LOAD_A]](%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[BOUND_ARRAY:.*]] = fir.alloca !fir.array<2xi64>
! CHECK: %[[ARRAY:.*]] = fir.undefined !fir.array<2xi64>
! CHECK: %[[ARRAY0:.*]] = fir.insert_value %[[ARRAY]], %[[C0]], [0 : index] : (!fir.array<2xi64>, i64) -> !fir.array<2xi64>
! CHECK: %[[ARRAY1:.*]] = fir.insert_value %[[ARRAY0]], %[[C99]], [1 : index] : (!fir.array<2xi64>, i64) -> !fir.array<2xi64>
! CHECK: fir.store %[[ARRAY1]] to %[[BOUND_ARRAY]] : !fir.ref<!fir.array<2xi64>>
! CHECK: %[[C2:.*]] = arith.constant 2 : index
! CHECK: %[[BOUND_ARRAY_SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK: %[[BOXED_BOUND_ARRAY:.*]] = fir.embox %[[BOUND_ARRAY]](%[[BOUND_ARRAY_SHAPE]]) : (!fir.ref<!fir.array<2xi64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi64>>
! CHECK: %[[ARG0:.*]] = fir.convert %[[Q]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[ARG1:.*]] = fir.convert %[[REBOX_A]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %[[ARG2:.*]] = fir.convert %[[BOXED_BOUND_ARRAY]] : (!fir.box<!fir.array<2xi64>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAssociateRemapping(%[[ARG0]], %[[ARG1]], %[[ARG2]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none

  subroutine test_elemental_assign()
    type(p1) :: pa(3)
    pa = [ 1, 2, 3 ]
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_assign() {
! CHECK: %[[INT:.*]] = fir.alloca i32
! CHECK: %[[C3_0:.*]] = arith.constant 3 : index
! CHECK: %[[PA:.*]] = fir.alloca !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "pa", uniq_name = "_QMpolymorphic_testFtest_elemental_assignEpa"}
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C3_0]] : (index) -> !fir.shape<1>
! CHECK: %[[LOAD_PA:.*]] = fir.array_load %[[PA]](%[[SHAPE]]) : (!fir.ref<!fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: %[[ADDR_INT:.*]] = fir.address_of(@_QQro.3xi4.{{.*}}) : !fir.ref<!fir.array<3xi32>>
! CHECK: %[[C3:.*]] = arith.constant 3 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C3]] : (index) -> !fir.shape<1>
! CHECK: %[[LOAD_INT_ARRAY:.*]] = fir.array_load %[[ADDR_INT]](%[[SHAPE]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C3_0]], %[[C1]] : index
! CHECK: %[[DO_RES:.*]] = fir.do_loop %[[ARG0:.*]] = %[[C0]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG1:.*]] = %[[LOAD_PA]]) -> (!fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {
! CHECK:   %[[FETCH_INT:.*]] = fir.array_fetch %[[LOAD_INT_ARRAY]], %[[ARG0]] : (!fir.array<3xi32>, index) -> i32
! CHECK:   %[[ARRAY_MOD:.*]]:2 = fir.array_modify %[[ARG1]], %[[ARG0]] : (!fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, index) -> (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>)
! CHECK:   %[[EMBOXED:.*]] = fir.embox %10#0 : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.store %[[FETCH_INT]] to %[[INT]] : !fir.ref<i32>
! CHECK:   fir.call @_QMpolymorphic_testPassign_p1_int(%[[EMBOXED]], %[[INT]]) fastmath<contract> : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.ref<i32>) -> ()
! CHECK:   fir.result %[[ARRAY_MOD]]#1 : !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: }
! CHECK: fir.array_merge_store %[[LOAD_PA]], %[[DO_RES]] to %[[PA]] : !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.ref<!fir.array<3x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: return

  subroutine host_assoc(this)
    class(p1) :: this
    
    call internal
  contains
    subroutine internal
      print*, this%a, this%b
    end subroutine
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testFhost_assocPinternal(
! CHECK-SAME: %[[TUPLE:.*]]: !fir.ref<tuple<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
! CHECK: %[[POS_IN_TUPLE:.*]] = arith.constant 0 : i32
! CHECK: %[[COORD_OF_CLASS:.*]] = fir.coordinate_of %[[TUPLE]], %[[POS_IN_TUPLE]] : (!fir.ref<tuple<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, i32) -> !fir.ref<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[CLASS:.*]] = fir.load %[[COORD_OF_CLASS]] : !fir.ref<!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK: %[[FIELD_A:.*]] = fir.field_index a, !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>
! CHECK: %[[COORD_A:.*]] = fir.coordinate_of %[[CLASS]], %[[FIELD_A]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK: %[[A:.*]] = fir.load %[[COORD_A]] : !fir.ref<i32>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[A]]) {{.*}} : (!fir.ref<i8>, i32) -> i1
! CHECK: %[[FIELD_B:.*]] = fir.field_index b, !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>
! CHECK: %[[COORD_B:.*]] = fir.coordinate_of %[[CLASS]], %[[FIELD_B]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK: %[[B:.*]] = fir.load %[[COORD_B]] : !fir.ref<i32>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[B]]) {{.*}} : (!fir.ref<i8>, i32) -> i1

  subroutine test_elemental_array()
    type(p1) :: p(5)
    print *, p%elemental_fct()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_array() {
! CHECK: %[[P:.*]] = fir.alloca !fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFtest_elemental_arrayEp"}
! CHECK: %[[C5:.*]] = arith.constant 5 : index
! CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<5xi32>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
! CHECK: %[[ARRAY_LOAD_TMP:.*]] = fir.array_load %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C5]], %[[C1]] : index
! CHECK: %[[LOOP_RES:.*]] = fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG1:.*]] = %[[ARRAY_LOAD_TMP]]) -> (!fir.array<5xi32>) {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.ref<!fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[RES:.*]] = fir.call @_QMpolymorphic_testPelemental_fct(%[[EMBOXED]]) {{.*}} : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> i32
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG1]], %[[RES]], %[[IND]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<5xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD_TMP]], %[[LOOP_RES]] to %[[TMP]] : !fir.array<5xi32>, !fir.array<5xi32>, !fir.heap<!fir.array<5xi32>>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOXED_TMP:.*]] = fir.embox %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOXED_TMP]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK: fir.freemem %[[TMP]] : !fir.heap<!fir.array<5xi32>>

  subroutine test_elemental_poly_array(p)
    class(p1) :: p(5)
    print *, p%elemental_fct()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_poly_array(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}) {
! CHECK: %[[C5:.*]] = arith.constant 5 : index
! CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<5xi32>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
! CHECK: %[[ARRAY_LOAD_TMP:.*]] = fir.array_load %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C5]], %[[C1]] : index
! CHECK: %[[LOOP_RES:.*]] = fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD_TMP]]) -> (!fir.array<5xi32>) {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.class<!fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[RES:.*]] = fir.dispatch "elemental_fct"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> i32 {pass_arg_pos = 0 : i32}
! CHECK:   %[[ARR_UP:.*]] = fir.array_update %[[ARG]], %[[RES]], %[[IND]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
! CHECK:   fir.result %[[ARR_UP]] : !fir.array<5xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD_TMP]], %[[LOOP_RES]] to %[[TMP]] : !fir.array<5xi32>, !fir.array<5xi32>, !fir.heap<!fir.array<5xi32>>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOXED_TMP:.*]] = fir.embox %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOXED_TMP]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK: fir.freemem %[[TMP]] : !fir.heap<!fir.array<5xi32>>

  subroutine test_elemental_poly_array_2d(p)
    class(p1) :: p(5,5)
    print *, p%elemental_fct()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_poly_array_2d(
! CHECK-SAME: %[[P]]: !fir.class<!fir.array<5x5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}) {
! CHECK: %[[C5:.*]] = arith.constant 5 : index
! CHECK: %[[C5_0:.*]] = arith.constant 5 : index
! CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<5x5xi32>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]], %[[C5_0]] : (index, index) -> !fir.shape<2>
! CHECK: %[[ARRAY_LOAD_TMP:.*]] = fir.array_load %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5x5xi32>>, !fir.shape<2>) -> !fir.array<5x5xi32>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB0:.*]] = arith.subi %[[C5]], %[[C1]] : index
! CHECK: %[[UB1:.*]] = arith.subi %[[C5_0]], %[[C1]] : index
! CHECK: %[[LOOP_RES:.*]] = fir.do_loop %[[IND0:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] unordered iter_args(%[[ARG:.*]] = %[[ARRAY_LOAD_TMP]]) -> (!fir.array<5x5xi32>) {
! CHECK:   %[[LOOP_RES0:.*]] = fir.do_loop %[[IND1:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] unordered iter_args(%[[ARG0:.*]] = %[[ARG]]) -> (!fir.array<5x5xi32>) {
! CHECK:     %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND1]], %[[IND0]] : (!fir.class<!fir.array<5x5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:     %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<5x5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:     %[[RES:.*]] = fir.dispatch "elemental_fct"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> i32 {pass_arg_pos = 0 : i32}
! CHECK:     %[[ARR_UP:.*]] = fir.array_update %[[ARG0]], %[[RES]], %[[IND1]], %[[IND0]] : (!fir.array<5x5xi32>, i32, index, index) -> !fir.array<5x5xi32>
! CHECK:     fir.result %[[ARR_UP]] : !fir.array<5x5xi32>
! CHECK:   }
! CHECK:   fir.result %[[LOOP_RES0]] : !fir.array<5x5xi32>
! CHECK: }
! CHECK: fir.array_merge_store %[[ARRAY_LOAD_TMP]], %[[LOOP_RES]] to %[[TMP]] : !fir.array<5x5xi32>, !fir.array<5x5xi32>, !fir.heap<!fir.array<5x5xi32>>
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C5]], %[[C5_0]] : (index, index) -> !fir.shape<2>
! CHECK: %[[EMBOXED_TMP:.*]] = fir.embox %[[TMP]](%[[SHAPE]]) : (!fir.heap<!fir.array<5x5xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<5x5xi32>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[EMBOXED_TMP]] : (!fir.box<!fir.array<5x5xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK: fir.freemem %[[TMP]] : !fir.heap<!fir.array<5x5xi32>>

  subroutine test_elemental_sub_array()
    type(p1) :: t(10)
    call t%elemental_sub()
    call t%elemental_sub_pass(2)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_array() {
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[T:.*]] = fir.alloca !fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "t", uniq_name = "_QMpolymorphic_testFtest_elemental_sub_arrayEt"}
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[T]], %[[IND]] : (!fir.ref<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.call @_QMpolymorphic_testPelemental_sub(%[[EMBOXED]]) {{.*}} : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: }
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[T]], %[[IND]] : (!fir.ref<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.call @_QMpolymorphic_testPelemental_sub_pass(%{{.*}}, %[[EMBOXED]]) {{.*}} : (!fir.ref<i32>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: }

  subroutine test_elemental_sub_poly_array(p)
    class(p1) :: p(10)
    call p%elemental_sub()
    call p%elemental_sub_pass(3)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_poly_array(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}) {
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.dispatch "elemental_sub"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK: }
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.dispatch "elemental_sub_pass"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%{{.*}}, %[[EMBOXED]] : !fir.ref<i32>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
! CHECK: }

  subroutine test_elemental_sub_array_assumed(t)
    type(p1) :: t(:)
    call t%elemental_sub()
    call t%elemental_sub_pass(4)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_array_assumed(
! CHECK-SAME: %[[T:.*]]: !fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "t"}) {
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[T_DIMS:.*]]:3 = fir.box_dims %[[T]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[T_DIMS]]#1, %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:  %[[COORD:.*]] = fir.coordinate_of %[[T]], %[[IND]] : (!fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[EMBOXED:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMpolymorphic_testPelemental_sub(%[[EMBOXED]]) {{.*}} : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[T_DIMS:.*]]:3 = fir.box_dims %[[T]], %[[C0]] : (!fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[T_DIMS]]#1, %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:  %[[COORD:.*]] = fir.coordinate_of %[[T]], %[[IND]] : (!fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  %[[EMBOXED:.*]] = fir.embox %[[COORD]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:  fir.call @_QMpolymorphic_testPelemental_sub_pass(%{{.*}}, %[[EMBOXED]]) {{.*}} : (!fir.ref<i32>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: }

  subroutine test_elemental_sub_poly_array_assumed(p)
    class(p1) :: p(:)
    call p%elemental_sub()
    call p%elemental_sub_pass(5)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_poly_array_assumed(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}) {
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[P_DIMS:.*]]:3 = fir.box_dims %[[P]], %[[C0]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[P_DIMS]]#1, %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.dispatch "elemental_sub"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
! CHECK: }
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[P_DIMS:.*]]:3 = fir.box_dims %[[P]], %[[C0]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[P_DIMS]]#1, %[[C1]] : index
! CHECK: fir.do_loop %[[IND:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
! CHECK:   %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[IND]] : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>, index) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD]] source_box %[[P]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   fir.dispatch "elemental_sub_pass"(%[[EMBOXED]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%{{.*}}, %[[EMBOXED]] : !fir.ref<i32>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
! CHECK: }

  subroutine write_p1(dtv, unit, iotype, v_list, iostat, iomsg)
    class(p1), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! dummy subroutine for testing purpose
  end subroutine

  subroutine read_p1(dtv, unit, iotype, v_list, iostat, iomsg)
    class(p1), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! dummy subroutine for testing purpose
  end subroutine

  subroutine test_polymorphic_io()
    type(p1), target :: t
    class(p1), pointer :: p
    open(17, form='formatted', access='stream')
    write(17, 1) t
    1 Format(1X,I10)
    p => t
    rewind(17)
    read(17, 1) p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_polymorphic_io() {
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFtest_polymorphic_ioEp"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[LOAD_P]] : (!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAioInputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}} : (!fir.ref<i8>, !fir.box<none>) -> i1

  function unlimited_polymorphic_alloc_array_ret()
    class(*), allocatable :: unlimited_polymorphic_alloc_array_ret(:)
  end function

  subroutine test_unlimited_polymorphic_alloc_array_ret()
    select type (a => unlimited_polymorphic_alloc_array_ret())
      type is (real)
        print*, 'type is real' 
    end select
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_unlimited_polymorphic_alloc_array_ret() {
! CHECK: %[[RES_TMP:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>> {bindc_name = ".result"}
! CHECK: %[[RES:.*]] = fir.call @_QMpolymorphic_testPunlimited_polymorphic_alloc_array_ret() fastmath<contract> : () -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: fir.save_result %[[RES]] to %[[RES_TMP]] : !fir.class<!fir.heap<!fir.array<?xnone>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>

  subroutine test_unlimited_polymorphic_intentout(a)
    class(*), intent(out) :: a
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_unlimited_polymorphic_intentout(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}) {
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAInitialize(%[[BOX_NONE]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32) -> none

  subroutine test_polymorphic_intentout(a)
    class(p1), intent(out) :: a
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_polymorphic_intentout(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAInitialize(%[[BOX_NONE]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32) -> none

  subroutine rebox_up_to_record_type(p)
    class(*), allocatable, target :: p(:,:)
    type(non_extensible), pointer :: t(:,:)
    t => p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPrebox_up_to_record_type(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>> {fir.bindc_name = "p", fir.target}) {
! CHECK: %[[T:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QMpolymorphic_testTnon_extensible{d:i32}>>>> {bindc_name = "t", uniq_name = "_QMpolymorphic_testFrebox_up_to_record_typeEt"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[LOAD_P]](%{{.*}}) : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>, !fir.shift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QMpolymorphic_testTnon_extensible{d:i32}>>>>
! CHECK: fir.store %[[REBOX]] to %[[T]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QMpolymorphic_testTnon_extensible{d:i32}>>>>>

  subroutine sub_with_poly_optional(a)
    class(*), optional :: a
  end subroutine

  subroutine test_call_with_null()
    call sub_with_poly_optional(null())
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_call_with_null() {
! CHECK: %[[NULL_LLVM_PTR:.*]] = fir.alloca !fir.llvm_ptr<none>
! CHECK: %[[NULL_BOX_NONE:.*]] = fir.convert %[[NULL_LLVM_PTR]] : (!fir.ref<!fir.llvm_ptr<none>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[BOX_NONE:.*]] = fir.load %[[NULL_BOX_NONE]] : !fir.ref<!fir.box<none>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_NONE]] : (!fir.box<none>) -> !fir.ref<none>
! CHECK: %[[BOX_ADDR_I64:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED_OR_ASSOCIATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_I64]], %[[C0]] : i64
! CHECK: %[[ABSENT:.*]] = fir.absent !fir.class<none>
! CHECK: %[[BOX_NONE:.*]] = fir.load %[[NULL_BOX_NONE]] : !fir.ref<!fir.box<none>>
! CHECK: %[[CLASS_NONE:.*]] = fir.convert %[[BOX_NONE]] : (!fir.box<none>) -> !fir.class<none>
! CHECK: %[[ARG:.*]] = arith.select %[[IS_ALLOCATED_OR_ASSOCIATED]], %[[CLASS_NONE]], %[[ABSENT]] : !fir.class<none>
! CHECK: fir.call @_QMpolymorphic_testPsub_with_poly_optional(%[[ARG]]) {{.*}} : (!fir.class<none>) -> ()

  subroutine sub_with_poly_array_optional(a)
    class(*), optional :: a(:)
  end subroutine

  subroutine test_call_with_pointer_to_optional()
    real, pointer :: p(:)
    call sub_with_poly_array_optional(p)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_call_with_pointer_to_optional() {
! CHECK: %[[P:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p", uniq_name = "_QMpolymorphic_testFtest_call_with_pointer_to_optionalEp"}
! CHECK: %[[IS_ALLOCATED_OR_ASSOCIATED:.*]] = arith.cmpi ne
! CHECK: %[[ABSENT:.*]] = fir.absent !fir.class<!fir.array<?xnone>>
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[LOAD_P]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK: %[[ARG:.*]] = arith.select %[[IS_ALLOCATED_OR_ASSOCIATED]], %[[REBOX]], %[[ABSENT]] : !fir.class<!fir.array<?xnone>>
! CHECK: fir.call @_QMpolymorphic_testPsub_with_poly_array_optional(%[[ARG]]) {{.*}} : (!fir.class<!fir.array<?xnone>>) -> ()

  subroutine sub_with_real_pointer_optional(p)
    real, optional :: p(:)
    call sub_with_poly_array_optional(p)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_real_pointer_optional(
! CHECK-SAME: %[[P:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "p", fir.optional}) {
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[P]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK: %[[BOX:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.class<!fir.array<?xnone>>) {
! CHECK:   %[[REBOX:.*]] = fir.rebox %[[P]] : (!fir.box<!fir.array<?xf32>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:   fir.result %[[REBOX]] : !fir.class<!fir.array<?xnone>>
! CHECK: } else {
! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.class<!fir.array<?xnone>>
! CHECK:   fir.result %[[ABSENT]] : !fir.class<!fir.array<?xnone>>
! CHECK: }
! CHECK: fir.call @_QMpolymorphic_testPsub_with_poly_array_optional(%[[BOX]]) {{.*}} : (!fir.class<!fir.array<?xnone>>) -> ()

end module

program test
  use polymorphic_test
  type(outer), allocatable :: o
  integer :: i(5)
  logical :: l(5)
  allocate(o)

  l = i < o%inner
end program

! CHECK-LABEL: func.func @_QQmain() {
! CHECK: %[[ADDR_O:.*]] = fir.address_of(@_QFEo) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMpolymorphic_testTouter{inner:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>}>>>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ADDR_O]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMpolymorphic_testTouter{inner:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: %[[O:.*]] = fir.load %[[ADDR_O]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMpolymorphic_testTouter{inner:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>}>>>>
! CHECK: %[[FIELD_INNER:.*]] = fir.field_index inner, !fir.type<_QMpolymorphic_testTouter{inner:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>}>
! CHECK: %[[COORD_INNER:.*]] = fir.coordinate_of %[[O]], %[[FIELD_INNER]] : (!fir.box<!fir.heap<!fir.type<_QMpolymorphic_testTouter{inner:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>}>>>, !fir.field) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: %{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%arg1 = %9) -> (!fir.array<5x!fir.logical<4>>) {
! CHECK:   %[[EMBOXED:.*]] = fir.embox %[[COORD_INNER]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:   %{{.*}} = fir.call @_QMpolymorphic_testPlt(%17, %[[EMBOXED]]) {{.*}} : (!fir.ref<i32>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.logical<4>
! CHECK:  }
