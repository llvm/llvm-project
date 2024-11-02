! RUN: bbc -polymorphic-type -emit-fir %s -o - | tco | FileCheck %s
! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s --check-prefix=BT

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
  end type

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

  subroutine proc_pass_p1(i, this)
    integer :: i
    class(p1) :: this
    print*, 'call proc_nopass_p1'
  end subroutine

  subroutine proc_pass_p2(i, this)
    integer :: i
    class(p2) :: this
    print*, 'call proc_nopass_p2'
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

end module

program test_type_to_class
  use dispatch1
  type(p1) :: t1 = p1(1,2)
  type(p2) :: t2 = p2(1,2,3)

  call display_class(t1)
  call display_class(t2)
end


! CHECK-LABEL: define void @_QMdispatch1Pdisplay_class(
! CHECK-SAME: ptr %[[CLASS:.*]])

! CHECK-DAG: %[[INT32:.*]] = alloca i32, i64 1
! CHECK-DAG: %[[REAL:.*]] = alloca float, i64 1
! CHECK-DAG: %[[I:.*]] = alloca i32, i64 1

! Check dynamic dispatch equal to `call p%display2()` with binding index = 2.
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 2
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]](ptr %[[CLASS]])

! Check dynamic dispatch equal to `call p%display1()` with binding index = 1.
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 1
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]](ptr %[[CLASS]])

! Check dynamic dispatch equal to `call p%aproc()` with binding index = 0.
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 0
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]](ptr %[[CLASS]])

! Check dynamic dispatch of a function with result.
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 3
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: %[[RET:.*]] = call i32 %[[FUNC_PTR]](ptr %[[CLASS]])
! CHECK: store i32 %[[RET]], ptr %[[I]]

! Check dynamic dispatch of call with passed-object and additional argument
! CHECK: store float 2.500000e+00, ptr %[[REAL]]
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 6
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]](ptr %[[CLASS]], ptr %[[REAL]])

! Check dynamic dispatch of a call with NOPASS
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 4
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]]()

! CHECK: store i32 1, ptr %[[INT32]]
! CHECK: %[[LOADED_CLASS:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS]]
! CHECK: %[[TYPEDESCPTR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED_CLASS]], 7
! CHECK: %[[LOADED_TYPEDESC:.*]] = load %_QM__fortran_type_infoTderivedtype, ptr %[[TYPEDESCPTR]]
! CHECK: %[[DT:.*]] = extractvalue %_QM__fortran_type_infoTderivedtype %[[LOADED_TYPEDESC]], 0
! CHECK: %[[BINDING_BASE_ADDR:.*]] = extractvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[DT]], 0
! CHECK: %[[BINDING_PTR:.*]] = getelementptr %_QM__fortran_type_infoTbinding, ptr %[[BINDING_BASE_ADDR]], i32 5
! CHECK: %[[LOADED_BINDING:.*]] = load %_QM__fortran_type_infoTbinding, ptr %[[BINDING_PTR]]
! CHECK: %[[BUILTIN_FUNC_PTR:.*]] = extractvalue %_QM__fortran_type_infoTbinding %[[LOADED_BINDING]], 0
! CHECK: %[[FUNC_ADDR:.*]] = extractvalue %_QM__fortran_builtinsT__builtin_c_funptr %[[BUILTIN_FUNC_PTR]], 0
! CHECK: %[[FUNC_PTR:.*]] = inttoptr i64 %[[FUNC_ADDR]] to ptr
! CHECK: call void %[[FUNC_PTR]](ptr %[[INT32]], ptr %[[CLASS]])

! CHECK-LABEL: _QMdispatch1Pno_pass_array
! CHECK-LABEL: _QMdispatch1Pno_pass_array_allocatable
! CHECK-LABEL: _QMdispatch1Pno_pass_array_pointer

! Check the layout of the binding table. This is easier to do in FIR than in 
! LLVM IR.

! BT-LABEL: fir.global linkonce_odr @_QMdispatch1E.v.p1 constant target : !fir.array<7x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>> {
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Paproc) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pdisplay1_p1) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pdisplay2_p1) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pget_value_p1) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> i32
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_nopass_p1) : () -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_pass_p1) : (!fir.ref<i32>, !fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_p1) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>, !fir.ref<f32>) -> ()
! BT: }

! BT-LABEL: fir.global linkonce_odr @_QMdispatch1E.v.p2 constant target : !fir.array<8x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>> {
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Paproc) : (!fir.class<!fir.type<_QMdispatch1Tp1{a:i32,b:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pdisplay1_p2) : (!fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pdisplay2_p2) : (!fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pget_value_p2) : (!fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>) -> i32
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_nopass_p2) : () -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_pass_p2) : (!fir.ref<i32>, !fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pproc_p2) : (!fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>, !fir.ref<f32>) -> ()
! BT: %{{.*}} = fir.address_of(@_QMdispatch1Pdisplay3) : (!fir.class<!fir.type<_QMdispatch1Tp2{a:i32,b:i32,c:i32}>>) -> ()
! BT: }

