! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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

  type :: p4
    class(p1), allocatable :: a(:)
  end type

  type :: p5
    class(*), allocatable :: up
  end type

  contains

  elemental subroutine assign_p1_int(lhs, rhs)
    class(p1), intent(inout) :: lhs
    integer, intent(in) :: rhs
    lhs%a = rhs
    lhs%b = rhs
  End Subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPassign_p1_int(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "lhs"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "rhs"}
! CHECK-SAME:    attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}
! CHECK:         %[[LHS:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMpolymorphic_testFassign_p1_intElhs"}
! CHECK:         %[[RHS:.*]]:2 = hlfir.declare %[[ARG1]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMpolymorphic_testFassign_p1_intErhs"}
! CHECK:         %[[A:.*]] = hlfir.designate %[[LHS]]#0{"a"}
! CHECK:         hlfir.assign %{{.*}} to %[[A]]
! CHECK:         %[[B:.*]] = hlfir.designate %[[LHS]]#0{"b"}
! CHECK:         hlfir.assign %{{.*}} to %[[B]]

  elemental integer function elemental_fct(this)
    class(p1), intent(in) :: this
    elemental_fct = this%a
  end function
! CHECK-LABEL: func.func @_QMpolymorphic_testPelemental_fct(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "this"})
! CHECK-SAME:    -> i32 attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}
! CHECK:         %[[THIS:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMpolymorphic_testFelemental_fctEthis"}
! CHECK:         %[[A:.*]] = hlfir.designate %[[THIS]]#0{"a"}

  elemental subroutine elemental_sub(this)
    class(p1), intent(inout) :: this
    this%a = this%a * this%b
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPelemental_sub(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "this"})
! CHECK-SAME:    attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}
! CHECK:         %[[THIS:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         hlfir.designate %[[THIS]]#0{"a"}
! CHECK:         hlfir.designate %[[THIS]]#0{"b"}

  elemental subroutine elemental_sub_pass(c, this)
    integer, intent(in) :: c
    class(p1), intent(inout) :: this
    this%a = this%a * this%b + c
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPelemental_sub_pass(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "c"}, %[[ARG1:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "this"})
! CHECK-SAME:    attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}

  logical elemental function lt(i, poly)
    integer, intent(in) :: i
    class(p1), intent(in) :: poly
    lt = i < poly%a
  End Function
! CHECK-LABEL: func.func @_QMpolymorphic_testPlt(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}, %[[ARG1:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "poly"})
! CHECK-SAME:    -> !fir.logical<4> attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}
! CHECK:         %[[I:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMpolymorphic_testFltEi"}
! CHECK:         %[[POLY:.*]]:2 = hlfir.declare %[[ARG1]]{{.*}}{fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMpolymorphic_testFltEpoly"}
! CHECK:         %[[A:.*]] = hlfir.designate %[[POLY]]#0{"a"}

  ! Test correct access to polymorphic entity component.
  subroutine component_access(p)
    class(p1) :: p
    print*, p%a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPcomponent_access(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "p"})
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{uniq_name = "_QMpolymorphic_testFcomponent_accessEp"}
! CHECK:         %[[A:.*]] = hlfir.designate %[[P]]#0{"a"}   : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.ref<i32>
! CHECK:         %[[A_LD:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[A_LD]]){{.*}}: (!fir.ref<i8>, i32) -> i1


  subroutine print(this)
    class(p1) :: this
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPprint(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "this"}

  ! Test embox of fir.type to fir.class to be passed-object.
  subroutine check()
    type(p1) :: t1
    type(p2) :: t2
    call t1%print()
    call t2%print()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPcheck()
! CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFcheckEt1"}
! CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFcheckEt2"}
! CHECK:         %[[T1_BOX:.*]] = fir.embox %[[T1]]#0 : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         %[[T1_CLASS:.*]] = fir.convert %[[T1_BOX]] : (!fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         fir.call @_QMpolymorphic_testPprint(%[[T1_CLASS]])
! CHECK:         %[[T2_BOX:.*]] = fir.embox %[[T2]]#0
! CHECK:         %[[T2_CLASS:.*]] = fir.convert %[[T2_BOX]]{{.*}} -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         fir.call @_QMpolymorphic_testPprint(%[[T2_CLASS]])


  subroutine test_allocate_unlimited_polymorphic_non_derived()
    class(*), pointer :: u
    allocate(integer::u)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_allocate_unlimited_polymorphic_non_derived()
! CHECK:         %[[U:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFtest_allocate_unlimited_polymorphic_non_derivedEu"}
! CHECK:         fir.call @_FortranAPointerNullifyIntrinsic(
! CHECK:         %{{.*}} = fir.call @_FortranAPointerAllocate(


  function test_fct_ret_class()
    class(p1), pointer :: test_fct_ret_class
  end function
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_fct_ret_class() -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

  subroutine call_fct()
    class(p1), pointer :: p
    p => test_fct_ret_class()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_fct()
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFcall_fctEp"}
! CHECK:         %[[CALL:.*]] = fir.call @_QMpolymorphic_testPtest_fct_ret_class() {{.*}}: () -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK:         fir.store %{{.*}} to %[[P]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>



  subroutine implicit_loop_with_polymorphic()
    class(p1), allocatable :: p(:)
    allocate(p(3))
    p%a = [ 1, 2, 3 ]
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPimplicit_loop_with_polymorphic() {
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMpolymorphic_testFimplicit_loop_with_polymorphicEp"}
! CHECK:         %{{.*}} = fir.call @_FortranAAllocatableAllocate(
! CHECK:         %[[A:.*]] = hlfir.designate %{{.*}}{"a"}{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         hlfir.assign %{{.*}} to %[[A]]


  subroutine polymorphic_to_nonpolymorphic(p)
    class(p1), pointer :: p(:)
    type(p1), allocatable, target :: t(:)
    t = p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpolymorphic_to_nonpolymorphic
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         hlfir.assign

! Just checking that FIR is generated without error.

  subroutine nonpolymorphic_to_polymorphic(p, t)
    type p1
    end type
    type(p1), pointer :: p(:)
    class(p1), target :: t(:)
    p(0:1) => t
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPnonpolymorphic_to_polymorphic(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>>>> {fir.bindc_name = "p"}, %[[ARG1:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>> {fir.bindc_name = "t", fir.target}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFnonpolymorphic_to_polymorphicEp"}
! CHECK:         %[[T:.*]]:2 = hlfir.declare %[[ARG1]]{{.*}}{fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFnonpolymorphic_to_polymorphicEt"}
! CHECK:         %[[REBOX:.*]] = fir.rebox %[[T]]#0 : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>>>
! CHECK:         %[[SHIFT:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[REMAP:.*]] = fir.rebox %[[REBOX]](%[[SHIFT]]) : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testFnonpolymorphic_to_polymorphicTp1>>>>
! CHECK:         fir.store %[[REMAP]] to %[[P]]#0


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
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFcall_up_retEp"}
! CHECK:         %{{.*}} = fir.call @_QMpolymorphic_testPup_ret() {{.*}}: () -> !fir.class<!fir.ptr<!fir.array<?xnone>>>


  subroutine associate_up_pointer(r)
    class(r1) :: r
    class(*), pointer :: p(:)
    p => r%rp
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPassociate_up_pointer(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTr1{rp:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "r"})
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFassociate_up_pointerEp"}
! CHECK:         %[[R:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{uniq_name = "_QMpolymorphic_testFassociate_up_pointerEr"}
! CHECK:         %[[RP:.*]] = hlfir.designate %[[R]]#0{"rp"}{{.*}}{fortran_attrs = #fir.var_attrs<pointer>}
! CHECK:         %[[RP_LD:.*]] = fir.load %[[RP]]
! CHECK:         %[[REBOX:.*]] = fir.rebox %[[RP_LD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:         fir.store %[[REBOX]] to %[[P]]#0


! Test that the fir.dispatch operation is created with the correct pass object
! and the pass_arg_pos attribute is incremented correctly when character
! function result is added as argument.

  function get_tmp(this)
    class(c1) :: this
    character(2) :: get_tmp
    get_tmp = this%tmp
  end function
! CHECK-LABEL: func.func @_QMpolymorphic_testPget_tmp(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.char<1,2>>
! CHECK-SAME:    %[[ARG1:.*]]: index{{.*}}, %[[ARG2:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTc1{tmp:!fir.char<1,2>}>> {fir.bindc_name = "this"}
! CHECK:         %[[THIS:.*]]:2 = hlfir.declare %[[ARG2]]{{.*}}{uniq_name = "_QMpolymorphic_testFget_tmpEthis"}
! CHECK:         %[[TMP:.*]] = hlfir.designate %[[THIS]]#0{"tmp"}

  subroutine call_get_tmp(c)
    class(c1) :: c
    print*, c%get_tmp()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_get_tmp(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTc1{tmp:!fir.char<1,2>}>> {fir.bindc_name = "c"})
! CHECK:         %[[C:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{uniq_name = "_QMpolymorphic_testFcall_get_tmpEc"}
! CHECK:         %{{.*}} = fir.dispatch "get_tmp"(%[[C]]#0


  subroutine sub_with_type_array(a)
    type(p1) :: a(:)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_type_array(%{{.*}}: !fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"})

  subroutine call_sub_with_type_array(p)
    class(p1), pointer :: p(:)
    call sub_with_type_array(p)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPcall_sub_with_type_array(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"})
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[P_LD:.*]] = fir.load %[[P]]#0
! CHECK:         %[[REBOX:.*]] = fir.rebox %[[P_LD]]
! CHECK:         fir.call @_QMpolymorphic_testPsub_with_type_array(%{{.*}})


  subroutine derived_type_assignment_with_class()
    type(p3) :: a
    type(p3), target :: b(10)
    a = p3(b)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPderived_type_assignment_with_class()
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFderived_type_assignment_with_classEa"}
! CHECK:         %[[B:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFderived_type_assignment_with_classEb"}
! CHECK:         %[[CTOR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "ctor.temp"}
! CHECK:         hlfir.assign %[[CTOR]]#0 to %[[A]]#0

  subroutine takes_p1(p)
    class(p1), intent(in) :: p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtakes_p1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "p"}

! TODO: implement polymorphic temporary in lowering
!  subroutine no_reassoc_poly_value(a, i)
!    class(p1), intent(in) :: a(:)
!    integer :: i
!    call takes_p1((a(i)))
!  end subroutine

! Test pointer assignment with non polymorphic lhs and polymorphic rhs

  subroutine pointer_assign_parent(p)
    type(p2), target :: p
    type(p1), pointer :: tp
    tp => p%p1
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_parent(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp2{p1:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>,c:f32}>> {{{.*}}, fir.target}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFpointer_assign_parentEp"}
! CHECK:         %[[TP:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_parentEtp"}
! CHECK:         %[[P1:.*]] = hlfir.designate %[[P]]#0{"p1"}
! CHECK:         %[[BOX:.*]] = fir.embox %[[P1]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.box<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>
! CHECK:         fir.store %[[BOX]] to %[[TP]]#0

! First test is here to have a reference with non polymorphic on both sides.

  subroutine pointer_assign_non_poly(p)
    class(p1), target :: p
    type(p1), pointer :: tp
    tp => p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_non_poly(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {{{.*}}, fir.target}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFpointer_assign_non_polyEp"}
! CHECK:         %[[TP:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_non_polyEtp"}
! CHECK:         %[[REBOX:.*]] = fir.rebox %[[P]]#0


  subroutine nullify_pointer_array(a)
    type(p3) :: a
    nullify(a%p)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPnullify_pointer_array(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp3>>>>}>> {fir.bindc_name = "a"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[P:.*]] = hlfir.designate %[[A]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<pointer>}
! CHECK:         fir.call @_FortranAPointerNullifyDerived(


  subroutine up_input(a)
    class(*), intent(in) :: a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPup_input(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}

  subroutine pass_trivial_to_up()
    call up_input('hello')
    call up_input(1)
    call up_input(2.5)
    call up_input(.true.)
    call up_input((-1.0,3))
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpass_trivial_to_up() {
! CHECK:         %[[STR_BOX:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>
! CHECK:         %[[STR_CLASS:.*]] = fir.rebox %[[STR_BOX]] : (!fir.box<!fir.char<1,5>>) -> !fir.class<none>
! CHECK:         fir.call @_QMpolymorphic_testPup_input(%[[STR_CLASS]])
! CHECK:         %[[I_BOX:.*]] = fir.embox %{{.*}} : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[I_CLASS:.*]] = fir.rebox %[[I_BOX]] : (!fir.box<i32>) -> !fir.class<none>
! CHECK:         fir.call @_QMpolymorphic_testPup_input(%[[I_CLASS]])






  subroutine up_arr_input(a)
    class(*), intent(in) :: a(2)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPup_arr_input(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<2xnone>> {fir.bindc_name = "a"}

  subroutine pass_trivial_arr_to_up()
    character :: c(2)
    integer :: i(2)
    real :: r(2)
    logical :: l(2)
    complex :: cx(2)

    call up_arr_input(c)
    call up_arr_input(i)
    call up_arr_input(r)
    call up_arr_input(l)
    call up_arr_input(cx)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpass_trivial_arr_to_up() {
! CHECK:         %{{.*}} = fir.embox %{{.*}}#0(%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1>>>
! CHECK:         %{{.*}} = fir.rebox %{{.*}} : (!fir.box<!fir.array<2x!fir.char<1>>>) -> !fir.class<!fir.array<2xnone>>
! CHECK:         fir.call @_QMpolymorphic_testPup_arr_input(%{{.*}})
! CHECK:         %{{.*}} = fir.embox %{{.*}}#0(%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:         %{{.*}} = fir.rebox %{{.*}} : (!fir.box<!fir.array<2xi32>>) -> !fir.class<!fir.array<2xnone>>






  subroutine assign_polymorphic_allocatable()
    type(p1), target :: t(10,20)
    class(p1), allocatable :: c(:,:)
    c = t
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPassign_polymorphic_allocatable() {
! CHECK:         %[[C:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMpolymorphic_testFassign_polymorphic_allocatableEc"}
! CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFassign_polymorphic_allocatableEt"}
! CHECK:         hlfir.assign %[[T]]#0 to %[[C]]#0 realloc


  subroutine pointer_assign_remap()
    class(p1), pointer :: a(:)
    class(p1), pointer :: p(:,:)
    class(p1), pointer :: q(:)
    allocate(a(100))
    p(1:10,1:10) => a
    q(0:99) => a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_remap() {
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_remapEa"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_remapEp"}
! CHECK:         %[[Q:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_remapEq"}


  subroutine pointer_assign_lower_bounds()
    class(p1), allocatable, target :: a(:)
    class(p1), pointer :: p(:)
    allocate(a(100))
    p(-50:) => a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpointer_assign_lower_bounds() {
! CHECK:         %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMpolymorphic_testFpointer_assign_lower_boundsEa"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFpointer_assign_lower_boundsEp"}


  subroutine test_elemental_assign()
    type(p1) :: pa(3)
    pa = [ 1, 2, 3 ]
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_assign() {
! CHECK:         %[[PA:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFtest_elemental_assignEpa"}
! CHECK:         hlfir.region_assign


  subroutine host_assoc(this)
    class(p1) :: this

    call internal
  contains
    subroutine internal
      print*, this%a, this%b
    end subroutine
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPhost_assoc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "this"}
! CHECK:         %[[THIS:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         fir.call @_QMpolymorphic_testFhost_assocPinternal(

! CHECK-LABEL: func.func private @_QMpolymorphic_testFhost_assocPinternal(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<tuple{{.*}}>{{.*}})
! CHECK:         %[[THIS_LD:.*]] = fir.load %{{.*}}
! CHECK:         %[[A:.*]] = hlfir.designate %{{.*}}{"a"}
! CHECK:         %[[B:.*]] = hlfir.designate %{{.*}}{"b"}


  subroutine test_elemental_array()
    type(p1) :: p(5)
    print *, p%elemental_fct()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_array() {
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFtest_elemental_arrayEp"}
! CHECK:         hlfir.elemental

  subroutine test_elemental_poly_array(p)
    class(p1) :: p(5)
    print *, p%elemental_fct()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_poly_array(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         hlfir.elemental

  subroutine test_elemental_poly_array_2d(p)
    class(p1) :: p(5,5)
    print *, p%elemental_fct()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_poly_array_2d(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<5x5x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]

  subroutine test_elemental_sub_array()
    type(p1) :: t(10)
    call t%elemental_sub()
    call t%elemental_sub_pass(2)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_array() {
! CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMpolymorphic_testFtest_elemental_sub_arrayEt"}

  subroutine test_elemental_sub_poly_array(p)
    class(p1) :: p(10)
    call p%elemental_sub()
    call p%elemental_sub_pass(3)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_poly_array(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<10x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}

  subroutine test_elemental_sub_array_assumed(t)
    type(p1) :: t(:)
    call t%elemental_sub()
    call t%elemental_sub_pass(4)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_array_assumed(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "t"}

  subroutine test_elemental_sub_poly_array_assumed(p)
    class(p1) :: p(:)
    call p%elemental_sub()
    call p%elemental_sub_pass(5)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_elemental_sub_poly_array_assumed(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "p"}


  subroutine write_p1(dtv, unit, iotype, v_list, iostat, iomsg)
    class(p1), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! dummy subroutine for testing purpose
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPwrite_p1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "dtv"}

  subroutine read_p1(dtv, unit, iotype, v_list, iostat, iomsg)
    class(p1), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! dummy subroutine for testing purpose
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPread_p1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "dtv"}

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
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFtest_polymorphic_ioEp"}
! CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMpolymorphic_testFtest_polymorphic_ioEt"}
! CHECK:         fir.call @_FortranAioOutputDerivedType(


  function unlimited_polymorphic_alloc_array_ret()
    class(*), allocatable :: unlimited_polymorphic_alloc_array_ret(:)
  end function
! CHECK-LABEL: func.func @_QMpolymorphic_testPunlimited_polymorphic_alloc_array_ret() -> !fir.class<!fir.heap<!fir.array<?xnone>>>

  subroutine test_unlimited_polymorphic_alloc_array_ret()
    select type (a => unlimited_polymorphic_alloc_array_ret())
      type is (real)
        print*, 'type is real'
    end select
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_unlimited_polymorphic_alloc_array_ret() {
! CHECK:         %{{.*}} = fir.call @_QMpolymorphic_testPunlimited_polymorphic_alloc_array_ret()
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.array<?xnone>>


  subroutine test_unlimited_polymorphic_intentout(a)
    class(*), intent(out) :: a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_unlimited_polymorphic_intentout(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}

  subroutine test_polymorphic_intentout(a)
    class(p1), intent(out) :: a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_polymorphic_intentout(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}

  subroutine rebox_up_to_record_type(p)
    class(*), allocatable, target :: p(:,:)
    type(non_extensible), pointer :: t(:,:)
    t => p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPrebox_up_to_record_type(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>> {{{.*}}, fir.target}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFrebox_up_to_record_typeEt"}


  subroutine sub_with_poly_optional(a)
    class(*), optional :: a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_poly_optional(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a", fir.optional}

  subroutine test_call_with_null()
    call sub_with_poly_optional(null())
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_call_with_null() {
! CHECK:         fir.call @_QMpolymorphic_testPsub_with_poly_optional(


  subroutine sub_with_poly_array_optional(a)
    class(*), optional :: a(:)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_poly_array_optional(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "a", fir.optional}

  subroutine test_call_with_pointer_to_optional()
    real, pointer :: p(:)
    call sub_with_poly_array_optional(p)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_call_with_pointer_to_optional() {
! CHECK:         %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMpolymorphic_testFtest_call_with_pointer_to_optionalEp"}
! CHECK:         fir.call @_QMpolymorphic_testPsub_with_poly_array_optional(


  subroutine sub_with_real_pointer_optional(p)
    real, optional :: p(:)
    call sub_with_poly_array_optional(p)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPsub_with_real_pointer_optional(

  subroutine pass_poly_pointer_optional(p)
    class(p1), pointer, optional :: p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpass_poly_pointer_optional(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>> {fir.bindc_name = "p", fir.optional}

  subroutine test_poly_pointer_null()
    call pass_poly_pointer_optional(null())
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_poly_pointer_null() {
! CHECK:         fir.call @_QMpolymorphic_testPpass_poly_pointer_optional(


  subroutine test_poly_array_component_output(p)
    class(p1), pointer :: p(:)
    print*, p(:)%a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_poly_array_component_output(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>> {fir.bindc_name = "p"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]


  subroutine opt_int(i)
    integer, optional, intent(in) :: i
    call opt_up(i)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPopt_int(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}

  subroutine opt_up(up)
    class(*), optional, intent(in) :: up
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPopt_up(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "up", fir.optional}

  function rhs()
    class(p1), pointer :: rhs
  end function
! CHECK-LABEL: func.func @_QMpolymorphic_testPrhs() -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

  subroutine test_rhs_assign(a)
    type(p1) :: a
    a = rhs()
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_rhs_assign(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]


  subroutine type_with_polymorphic_components(a, b)
    type(p4) :: a, b
    a = b
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtype_with_polymorphic_components(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp4{a:!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>}>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QMpolymorphic_testTp4{a:!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>}>> {fir.bindc_name = "b"}


  subroutine up_pointer(p)
    class(*), pointer, intent(in) :: p
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPup_pointer(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<none>>> {fir.bindc_name = "p"}

  subroutine test_char_to_up_pointer(c)
    character(*), target :: c
    call up_pointer(c)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_char_to_up_pointer(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c", fir.target}


  subroutine move_alloc_poly(a, b)
    class(p1), allocatable :: a, b

    call move_alloc(a, b)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPmove_alloc_poly(
! CHECK:         fir.call @_FortranAMoveAlloc(


  subroutine test_parent_comp_in_select_type(s)
    class(p1), allocatable :: s
    class(p1), allocatable :: p

    allocate(p1::p)

    select type(s)
      type is(p2)
        s%p1 = p
    end select
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_parent_comp_in_select_type(
! CHECK:         fir.select_type


  subroutine move_alloc_unlimited_poly(a, b)
    class(*), allocatable :: a, b

    call move_alloc(a, b)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPmove_alloc_unlimited_poly(
! CHECK:         fir.call @_FortranAMoveAlloc(


  subroutine test_parent_comp_intrinsic(a, b)
    class(p1) :: a
    type(p2), allocatable :: b
    logical :: c

    c = same_type_as(a, b%p1)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_parent_comp_intrinsic(
! CHECK:         fir.call @_FortranASameTypeAs(


  subroutine test_parent_comp_normal(a)
    class(p2) :: a

    call print(a%p1)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_parent_comp_normal(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp2{p1:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>,c:f32}>> {fir.bindc_name = "a"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{uniq_name = "_QMpolymorphic_testFtest_parent_comp_normalEa"}
! CHECK:         %[[P1:.*]] = hlfir.designate %[[A]]#0{"p1"}{{.*}} : (!fir.class<!fir.type<_QMpolymorphic_testTp2{p1:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>,c:f32}>>) -> !fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         %[[P1_BOX:.*]] = fir.embox %[[P1]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         %[[P1_CLASS:.*]] = fir.convert %[[P1_BOX]] : (!fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         fir.call @_QMpolymorphic_testPprint(%[[P1_CLASS]])


  subroutine takes_p1_opt(a)
    class(p1), optional :: a
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtakes_p1_opt(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a", fir.optional}

  subroutine test_parent_comp_opt(p)
    type(p2), allocatable :: p
    allocate(p)
    call takes_p1_opt(p%p1)
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPtest_parent_comp_opt(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMpolymorphic_testTp2{p1:!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>,c:f32}>>>> {fir.bindc_name = "p"}
! CHECK:         %[[P:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMpolymorphic_testFtest_parent_comp_optEp"}
! CHECK:         %{{.*}} = fir.call @_FortranAAllocatableAllocate(
! CHECK:         %[[P_LD:.*]] = fir.load %[[P]]#0
! CHECK:         %[[P_ADDR:.*]] = fir.box_addr %[[P_LD]]
! CHECK:         %[[P1:.*]] = hlfir.designate %[[P_ADDR]]{"p1"}
! CHECK:         %[[P1_BOX:.*]] = fir.embox %[[P1]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         %[[P1_CLASS:.*]] = fir.convert %[[P1_BOX]] : (!fir.box<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK:         fir.call @_QMpolymorphic_testPtakes_p1_opt(%[[P1_CLASS]])


  subroutine class_with_entry(a)
    class(p1) :: a,b
    select type (a)
    type is(p2)
      print*, a%c
    class default
      print*, a%a
    end select
    return
  entry d(b)
    select type(b)
    type is(p2)
      print*,b%c
    class default
      print*,b%a
    end select
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPclass_with_entry(
! CHECK-SAME:    %[[A:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"})
! CHECK:         %[[B:.*]] = fir.alloca !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "b", uniq_name = "_QMpolymorphic_testFclass_with_entryEb"}
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>

! CHECK-LABEL: func.func @_QMpolymorphic_testPd(
! CHECK-SAME:    %[[B:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "b"})
! CHECK:         %[[A:.*]] = fir.alloca !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {bindc_name = "a", uniq_name = "_QMpolymorphic_testFclass_with_entryEa"}
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>


  subroutine class_array_with_entry(a)
    class(p1) :: a(:), b(:)
    select type (a)
    type is(p2)
      print*, a%c
    class default
      print*, a%a
    end select
    return
  entry g(b)
    select type(b)
    type is(p2)
      print*,b%c
    class default
      print*,b%a
    end select
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPclass_array_with_entry(
! CHECK-SAME:    %[[A:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "a"})
! CHECK:         %[[B:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>

! CHECK-LABEL: func.func @_QMpolymorphic_testPg(
! CHECK-SAME:    %[[B:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>> {fir.bindc_name = "b"})
! CHECK:         %[[A:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>>
! CHECK:         fir.select_type %{{.*}} : !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>>


  subroutine pass_up(up)
    class(*), intent(in) :: up
  end subroutine
! CHECK-LABEL: func.func @_QMpolymorphic_testPpass_up(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "up"}

! TODO: unlimited polymorphic temporary in lowering
!  subroutine parenthesized_up(a)
!    type(p5) :: a
!    call pass_up((a%up))
!  end subroutine

end module

program test
  use polymorphic_test
  type(outer), allocatable :: o
  integer :: i(5)
  logical :: l(5)
  allocate(o)

  l = i < o%inner
end program

