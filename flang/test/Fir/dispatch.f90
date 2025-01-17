! RUN: bbc -emit-hlfir %s -o - | fir-opt --fir-polymorphic-op | FileCheck %s
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefix=BT

! Tests codegen of fir.dispatch operation. This test is intentionally run from
! Fortran through bbc and tco so we have all the binding tables lowered to FIR
! from semantics.

module dispatch1

  type p1
    integer :: a
    integer :: b
  contains
    procedure :: aproc
    procedure :: display1 => display1_p1
    procedure :: display2 => display2_p1
    procedure :: get_value => get_value_p1
    procedure :: proc_with_values => proc_p1
    procedure, nopass :: proc_nopass => proc_nopass_p1
    procedure, pass(this) :: proc_pass => proc_pass_p1
    procedure, nopass :: z_proc_nopass_bindc => proc_nopass_bindc_p1
  end type

  type, extends(p1) :: p2
    integer :: c
  contains
    procedure :: display1 => display1_p2
    procedure :: display2 => display2_p2
    procedure :: display3
    procedure :: get_value => get_value_p2
    procedure :: proc_with_values => proc_p2
    procedure, nopass :: proc_nopass => proc_nopass_p2
    procedure, pass(this) :: proc_pass => proc_pass_p2
    procedure, nopass :: z_proc_nopass_bindc => proc_nopass_bindc_p2
  end type

  type, abstract :: a1
    integer a
  contains
    procedure :: a1_proc
  end type

  type, extends(a1) :: a2
    integer b
  contains
    procedure :: a1_proc => a2_proc
  end type

  type ty_kind(i, j)
    integer, kind :: i, j
    integer :: a(i)
  end Type

  type, extends(ty_kind) :: ty_kind_ex
    integer :: b(j)
  end type
  type(ty_kind(10,20)) :: tk1
  type(ty_kind_ex(10,20)) :: tke1
contains

  subroutine display1_p1(this)
    class(p1) :: this
    print*,'call display1_p1'
  end subroutine

  subroutine display2_p1(this)
    class(p1) :: this
    print*,'call display2_p1'
  end subroutine

  subroutine display1_p2(this)
    class(p2) :: this
    print*,'call display1_p2'
  end subroutine

  subroutine display2_p2(this)
    class(p2) :: this
    print*,'call display2_p2'
  end subroutine

  subroutine aproc(this)
    class(p1) :: this
    print*,'call aproc'
  end subroutine

  subroutine display3(this)
    class(p2) :: this
    print*,'call display3'
  end subroutine

  function get_value_p1(this)
    class(p1) :: this
    integer :: get_value_p1
    get_value_p1 = 10
  end function

  function get_value_p2(this)
    class(p2) :: this
    integer :: get_value_p2
    get_value_p2 = 10
  end function

  subroutine proc_p1(this, v)
    class(p1) :: this
    real :: v
    print*, 'call proc1 with ', v
  end subroutine

  subroutine proc_p2(this, v)
    class(p2) :: this
    real :: v
    print*, 'call proc1 with ', v
  end subroutine

  subroutine proc_nopass_p1()
    print*, 'call proc_nopass_p1'
  end subroutine

  subroutine proc_nopass_p2()
    print*, 'call proc_nopass_p2'
  end subroutine

  subroutine proc_nopass_bindc_p1() bind(c)
    print*, 'call proc_nopass_bindc_p1'
  end subroutine

  subroutine proc_nopass_bindc_p2() bind(c)
    print*, 'call proc_nopass_bindc_p2'
  end subroutine

  subroutine proc_pass_p1(i, this)
    integer :: i
    class(p1) :: this
    print*, 'call proc_pass_p1'
  end subroutine

  subroutine proc_pass_p2(i, this)
    integer :: i
    class(p2) :: this
    print*, 'call proc_pass_p2'
  end subroutine

  subroutine display_class(p)
    class(p1) :: p
    integer :: i
    call p%display2()
    call p%display1()
    call p%aproc()
    i = p%get_value()
    call p%proc_with_values(2.5)
    call p%proc_nopass()
    call p%proc_pass(1)
    call p%z_proc_nopass_bindc()
  end subroutine

  subroutine no_pass_array(a)
    class(p1) :: a(:)
    call a(1)%proc_nopass()
  end subroutine

  subroutine no_pass_array_allocatable(a)
    class(p1), allocatable :: a(:)
    call a(1)%proc_nopass()
  end subroutine

  subroutine no_pass_array_pointer(a)
    class(p1), allocatable :: a(:)
    call a(1)%proc_nopass()
  end subroutine

  subroutine a1_proc(this)
    class(a1) :: this
  end subroutine

  subroutine a2_proc(this)
    class(a2) :: this
  end subroutine

  subroutine call_a1_proc(p)
    class(a1), pointer :: p
    call p%a1_proc()
  end subroutine

end module

program test_type_to_class
  use dispatch1
  type(p1) :: t1 = p1(1,2)
  type(p2) :: t2 = p2(1,2,3)

  call display_class(t1)
  call display_class(t2)
end


! CHECK-LABEL: func.func @_QMdispatch1Pdisplay_class(
! CHECK-SAME: %[[ARG:.*]]: [[CLASS:!fir.class<.*>>]]
! CHECK: %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMdispatch1Fdisplay_classEp"} : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>, !fir.dscope) -> (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>, !fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>)

! Check dynamic dispatch equal to `call p%display2()` with binding index = 2.
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c2{{.*}} : (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (([[CLASS]]) -> ())
! CHECK: fir.call %[[FUNC_PTR]](%[[ARG_DECL]]#0) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()

! Check dynamic dispatch equal to `call p%display1()` with binding index = 1.
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c1{{.*}} : (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (([[CLASS]]) -> ())
! CHECK: fir.call %[[FUNC_PTR]](%[[ARG_DECL]]#0) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()

! Check dynamic dispatch equal to `call p%aproc()` with binding index = 0.
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c0{{.*}}: (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (([[CLASS]]) -> ())
! CHECK: fir.call %[[FUNC_PTR]](%[[ARG_DECL]]#0) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()

! Check dynamic dispatch of a function with result.
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c3 : (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (([[CLASS]]) -> i32)
! CHECK: %[[RES:.*]] = fir.call %[[FUNC_PTR]](%[[ARG_DECL]]#0) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> i32

! Check dynamic dispatch of call with passed-object and additional argument
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c6{{.*}} : (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (([[CLASS]], !fir.ref<f32>) -> ())
! CHECK: fir.call %[[FUNC_PTR]](%[[ARG_DECL]]#0, %{{.*}}) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>, !fir.ref<f32>) -> ()

! Check dynamic dispatch of a call with NOPASS
! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#1 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>>>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c4{{.*}} :  (!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>, index) -> !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> (() -> ())
! CHECK: fir.call %[[FUNC_PTR]]() : () -> ()

! CHECK: %[[BOXDESC:.*]] = fir.box_tdesc %[[ARG_DECL]]#0 : ([[CLASS]]) -> !fir.tdesc<none>
! CHECK: %[[TYPEDESCPTR:.*]] = fir.convert %[[BOXDESC]] : (!fir.tdesc<none>) -> !fir.ref<[[TYPEINFO:!fir.type<_QM__fortran_type_infoTderivedtype{.*}>]]>
! CHECK: %[[BINDING_FIELD:.*]] = fir.field_index binding, [[TYPEINFO]]
! CHECK: %[[BINDING_BOX_ADDR:.*]] =  fir.coordinate_of %[[TYPEDESCPTR]], %[[BINDING_FIELD]] : (!fir.ref<[[TYPEINFO]]>, !fir.field) -> !fir.ref<[[BINDING_BOX_TYPE:.*]]>
! CHECK: %[[BINDING_BOX:.*]] = fir.load %[[BINDING_BOX_ADDR]] : !fir.ref<[[BINDING_BOX_TYPE]]>
! CHECK: %[[BINDING_BASE_ADDR:.*]] = fir.box_addr %[[BINDING_BOX]] : ([[BINDING_BOX_TYPE]]) -> !fir.ptr<[[BINDINGSINFO:.*]]>
! CHECK: %[[BINDING_PTR:.*]] = fir.coordinate_of %[[BINDING_BASE_ADDR]], %c5{{.*}} : (!fir.ptr<[[BINDINGSINFO]]>, index) -> !fir.ref<[[BINDINGINFO:.*]]>
! CHECK: %[[PROC_FIELD:.*]] = fir.field_index proc, [[BINDINGINFO]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = fir.coordinate_of %[[BINDING_PTR]], %[[PROC_FIELD]] : ({{.*}}) -> !fir.ref<[[BUILTIN_FUNC_TYPE:.*]]>
! CHECK: %[[ADDRESS_FIELD:.*]] = fir.field_index __address, [[BUILTIN_FUNC_TYPE]]
! CHECK: %[[FUNC_ADDR_PTR:.*]] = fir.coordinate_of %[[BUILTIN_FUNC_PTR]], %[[ADDRESS_FIELD]]
! CHECK: %[[FUNC_ADDR:.*]] = fir.load %[[FUNC_ADDR_PTR]] : !fir.ref<i64>
! CHECK: %[[FUNC_PTR:.*]] = fir.convert %[[FUNC_ADDR]] : (i64) -> ((!fir.ref<i32>, [[CLASS]]) -> ())
! CHECK: fir.call %[[FUNC_PTR]](%{{.*}}, %[[ARG_DECL]]#0) : (!fir.ref<i32>, [[CLASS]]) -> ()

! Test attributes are propagated from fir.dispatch to fir.call
! for `call p%z_proc_nopass_bindc()`
! CHECK: fir.call %{{.*}}() proc_attrs<bind_c> : () -> ()

! CHECK-LABEL: _QMdispatch1Pno_pass_array
! CHECK-LABEL: _QMdispatch1Pno_pass_array_allocatable
! CHECK-LABEL: _QMdispatch1Pno_pass_array_pointer
! CHECK-LABEL: _QMdispatch1Pcall_a1_proc

! Check the layout of the binding table. This is easier to do in FIR than in 
! LLVM IR.

! BT-LABEL: fir.type_info @_QMdispatch1Tty_kindK10K20
! BT-LABEL: fir.type_info @_QMdispatch1Tty_kind_exK10K20 {{.*}}extends !fir.type<_QMdispatch1Tty_kindK10K20{{.*}}>

! BT-LABEL: fir.type_info @_QMdispatch1Tp1
! BT: fir.dt_entry "aproc", @_QMdispatch1Paproc
! BT: fir.dt_entry "display1", @_QMdispatch1Pdisplay1_p1
! BT: fir.dt_entry "display2", @_QMdispatch1Pdisplay2_p1
! BT: fir.dt_entry "get_value", @_QMdispatch1Pget_value_p1
! BT: fir.dt_entry "proc_nopass", @_QMdispatch1Pproc_nopass_p1
! BT: fir.dt_entry "proc_pass", @_QMdispatch1Pproc_pass_p1
! BT: fir.dt_entry "proc_with_values", @_QMdispatch1Pproc_p1
! BT: fir.dt_entry "z_proc_nopass_bindc", @proc_nopass_bindc_p1
! BT: }

! BT-LABEL: fir.type_info @_QMdispatch1Ta1
! BT: fir.dt_entry "a1_proc", @_QMdispatch1Pa1_proc
! BT: }

! BT-LABEL: fir.type_info @_QMdispatch1Ta2 {{.*}}extends !fir.type<_QMdispatch1Ta1{{.*}}>
! BT:  fir.dt_entry "a1_proc", @_QMdispatch1Pa2_proc
! BT: }

! BT-LABEL: fir.type_info @_QMdispatch1Tp2 {{.*}}extends !fir.type<_QMdispatch1Tp1{{.*}}>
! BT:  fir.dt_entry "aproc", @_QMdispatch1Paproc
! BT:  fir.dt_entry "display1", @_QMdispatch1Pdisplay1_p2
! BT:  fir.dt_entry "display2", @_QMdispatch1Pdisplay2_p2
! BT:  fir.dt_entry "get_value", @_QMdispatch1Pget_value_p2
! BT:  fir.dt_entry "proc_nopass", @_QMdispatch1Pproc_nopass_p2
! BT:  fir.dt_entry "proc_pass", @_QMdispatch1Pproc_pass_p2
! BT:  fir.dt_entry "proc_with_values", @_QMdispatch1Pproc_p2
! BT:  fir.dt_entry "z_proc_nopass_bindc", @proc_nopass_bindc_p2
! BT:  fir.dt_entry "display3", @_QMdispatch1Pdisplay3
! BT: }
