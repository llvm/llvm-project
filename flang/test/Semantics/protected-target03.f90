! RUN: %python %S/test_errors.py %s %flang_fc1 -Wmismatching-dummy-procedure -Wexternal-interface-mismatch
! J3/26-007r1 C870-C874 propagation, procedure compatibility, the C864/NOTE 1
! conflict, and same-kind conversions of protected targets.

module propagation
  use iso_c_binding, only: c_f_pointer, c_null_ptr
  implicit none
  abstract interface
    subroutine callback()
    end
  end interface
  type plain
    integer, pointer :: q
  end type
  type protected
    integer, pointer, protected_target :: q
  end type
  type plain_array
    integer, pointer :: q(:)
  end type
  type protected_array
    integer, pointer, protected_target :: q(:)
  end type
  type plain_character
    character(2), pointer :: q
  end type
  type protected_character
    character(2), pointer, protected_target :: q
  end type
  type node
    integer :: value
    integer, pointer :: separate, separate_array(:)
    type(node), pointer :: next
    procedure(callback), pointer, nopass :: proc
  end type
  type container
    type(node), pointer, protected_target :: link
  end type
  integer, target :: x = 1
  type(node), target :: target_node
  abstract interface
    subroutine protected_interface(p)
      integer, pointer, protected_target, intent(in), optional :: p
    end
    subroutine ordinary_interface(p)
      integer, pointer, intent(in), optional :: p
    end
    function ordinary_result() result(r)
      integer, pointer :: r
    end
    function protected_result() result(r)
      integer, pointer, protected_target :: r
    end
  end interface
  interface generic_read
    module procedure data_in
  end interface
  interface generic_default
    module procedure data_default
  end interface
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => x
  end function
  function get_node() result(r)
    type(node), pointer, protected_target :: r
    r => target_node
  end function
  function from_argument(a) result(r)
    integer, intent(in) :: a
    integer, pointer, protected_target :: r
    r => x
  end function

  subroutine assignment(p, a, c, z, object, root)
    integer, pointer, protected_target :: p, a(:)
    character(8), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    type(node), pointer, protected_target :: object
    type(container) :: root
    integer, pointer :: w, wa(:)
    integer, pointer, protected_target :: keep, keep_array(:)
    character(2), pointer :: wc
    character(2), pointer, protected_target :: keep_character
    real, pointer :: wr
    type(plain) :: v
    type(protected) :: pv
    type(plain_array) :: av
    type(protected_array) :: pav
    type(plain_character) :: cv
    type(protected_character) :: pcv
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => p
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject may not be associated with pointer 'wa' without PROTECTED_TARGET
    wa => a(1:2)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => object%value
    !ERROR: PROTECTED_TARGET pointer 'link' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => root%link%value
    !ERROR: PROTECTED_TARGET pointer 'c' or its subobject may not be associated with pointer 'wc' without PROTECTED_TARGET
    wc => c(2:3)
    !ERROR: PROTECTED_TARGET pointer 'z' or its subobject may not be associated with pointer 'wr' without PROTECTED_TARGET
    wr => z%re
    !ERROR: pointer 'w' may not be associated with the PROTECTED_TARGET result of function 'get' without PROTECTED_TARGET
    w => get()
    keep => x
    keep => w
    keep => p
    keep => get()
    keep_array => a(1:2)
    keep_character => c(2:3)
    w => object%separate ! This denotes a different pointer's target.
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    v = plain(p)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    v = plain(object%value)
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    av = plain_array(a(1:2))
    !ERROR: PROTECTED_TARGET pointer 'c' or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    cv = plain_character(c(2:3))
    pv = protected(p)
    pav = protected_array(a(1:2))
    pcv = protected_character(c(2:3))
    associate (s => a(1:2))
      !ERROR: PROTECTED_TARGET pointer 'a' or its subobject may not be associated with pointer 'wa' without PROTECTED_TARGET
      wa => s
      keep_array => s
    end associate
    associate (value => get())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
      value = 1
      !ERROR: PROTECTED_TARGET pointer 'r' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
      w => value
      keep => value
    end associate
    associate (n => get_node())
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
      n%separate => x
      !ERROR: 'separate' may not appear in NULLIFY
      !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
      nullify(n%separate)
      n%separate = 1
    end associate
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
    get() = 1
  end subroutine

  subroutine calls(p, a, object, flag)
    integer, pointer, protected_target, optional :: p
    integer, pointer, protected_target :: a(:)
    type(node), pointer, protected_target :: object
    logical :: flag
    integer, pointer :: w
    integer :: value
    integer, external :: implicit_function
    external implicit_sub
    procedure(protected_interface), pointer :: pp
    procedure(protected_result), pointer :: pf
    procedure(ordinary_interface), pointer :: unprotected_procedure
    procedure(ordinary_result), pointer :: unprotected_function
    call pointer_in(p)
    call pointer_in(get())
    call pointer_out(p)
    call pointer_inout(p)
    call pointer_in(w)
    call pointer_array_in(a(1:2))
    call data_in(p)
    call data_in(object%value)
    call generic_read(p)
    call data_default(p + 1)
    call data_default((p))
    !ERROR: PROTECTED_TARGET pointer 'r' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_default(get())
    pp => pointer_in
    call pp(p)
    pf => get
    unprotected_procedure => pointer_in
    pp => unprotected_procedure
    unprotected_function => get
    !ERROR: pointer 'w' may not be associated with the PROTECTED_TARGET result of function 'pf' without PROTECTED_TARGET
    w => pf()
    w => null(p)
    print *, associated(p), present(p), size(a)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_in(p)
    !ERROR: dummy argument 'p=' may not be associated with the PROTECTED_TARGET result of function 'get' without PROTECTED_TARGET
    call ordinary_pointer_in(get())
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_in(object%value)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_out(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_inout(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_default(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_value(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call generic_default(p)
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_default(a(1))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
    call implicit_sub(p)
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject requires an explicit interface
    call implicit_sub(a(1:2))
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject requires an explicit interface
    value = implicit_function(object%value)
    call data_in((flag ? p : w))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'p='
    call data_default((flag ? p : w))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'fptr=' without PROTECTED_TARGET
    call c_f_pointer(c_null_ptr, p)
  end subroutine

  subroutine component_calls(object, objects, poly, whole_in)
    type(node), pointer, protected_target :: object, objects(:)
    class(node), pointer, protected_target :: poly
    integer, pointer, protected_target, intent(in) :: whole_in
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_in(object%separate)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_default(object%separate)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_out(object%separate)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_inout(object%separate)
    !ERROR: PROTECTED_TARGET pointer 'objects' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer_in(objects(1)%separate)
    call pointer_in(object%separate)
    call pointer_default(object%separate)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'p=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable via that pointer
    call pointer_out(object%separate)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable via that pointer
    call pointer_inout(object%separate)
    call data_in(object%separate)
    call data_default(object%separate)
    call data_out(object%separate)
    call data_inout(object%separate)
    call ordinary_pointer_array_in(object%separate_array(1:2))
    call ordinary_pointer_in(object%separate_array(1))
    call pointer_default(whole_in)
    call pointer_default(get())
    call ordinary_pointer_in(object%next%separate)
    call ordinary_pointer_default(object%next%separate)
    call ordinary_pointer_out(object%next%separate)
    call ordinary_pointer_inout(object%next%separate)
    associate (v => object)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_in(v%separate)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_default(v%separate)
      call pointer_default(v%separate)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_in(v%proc)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_default(v%proc)
      call procedure_target(v%proc)
    end associate
    select type (v => poly)
    type is (node)
      !ERROR: PROTECTED_TARGET pointer 'poly' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_in(v%separate)
      !ERROR: PROTECTED_TARGET pointer 'poly' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_default(v%separate)
      call pointer_default(v%separate)
      !ERROR: PROTECTED_TARGET pointer 'poly' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_in(v%proc)
      !ERROR: PROTECTED_TARGET pointer 'poly' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_default(v%proc)
      call procedure_target(v%proc)
    end select
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
    call procedure_pointer_in(object%proc)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
    call procedure_pointer_default(object%proc)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
    call procedure_pointer_out(object%proc)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
    call procedure_pointer_inout(object%proc)
    call procedure_target(object%proc)
    call procedure_pointer_in(object%next%proc)
    call procedure_pointer_default(object%next%proc)
    call procedure_pointer_out(object%next%proc)
    call procedure_pointer_inout(object%next%proc)
    call procedure_target(object%next%proc)
  end subroutine

  subroutine rank_component_calls(object)
    type(node), pointer, protected_target :: object(..)
    select rank (v => object)
    rank (0)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_in(v%separate)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_in(v%proc)
      call pointer_default(v%separate)
      call procedure_target(v%proc)
    rank (1)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_pointer_default(v(1)%separate)
      !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be associated with procedure pointer dummy argument 'p='
      call procedure_pointer_default(v(1)%proc)
      call pointer_default(v(1)%separate)
      call procedure_target(v(1)%proc)
    end select
  end subroutine

  subroutine pointer_in(p)
    integer, pointer, protected_target, intent(in), optional :: p
  end
  subroutine pointer_out(p)
    integer, pointer, protected_target, intent(out) :: p
    nullify(p)
  end
  subroutine pointer_inout(p)
    integer, pointer, protected_target, intent(inout) :: p
    nullify(p)
  end
  subroutine pointer_array_in(p)
    integer, pointer, protected_target, intent(in) :: p(:)
  end
  subroutine ordinary_pointer_in(p)
    integer, pointer, intent(in) :: p
  end
  subroutine ordinary_pointer_default(p)
    integer, pointer :: p
  end
  subroutine ordinary_pointer_out(p)
    integer, pointer, intent(out) :: p
  end
  subroutine ordinary_pointer_inout(p)
    integer, pointer, intent(inout) :: p
  end
  subroutine ordinary_pointer_array_in(p)
    integer, pointer, intent(in) :: p(:)
  end
  subroutine pointer_default(p)
    integer, pointer, protected_target :: p
    print *, associated(p)
  end
  subroutine procedure_pointer_in(p)
    procedure(callback), pointer, intent(in) :: p
  end
  subroutine procedure_pointer_default(p)
    procedure(callback), pointer :: p
  end
  subroutine procedure_pointer_out(p)
    procedure(callback), pointer, intent(out) :: p
  end
  subroutine procedure_pointer_inout(p)
    procedure(callback), pointer, intent(inout) :: p
  end
  subroutine procedure_target(p)
    procedure(callback) :: p
  end
  subroutine data_in(p)
    integer, intent(in) :: p
  end
  subroutine data_out(p)
    integer, intent(out) :: p
  end
  subroutine data_inout(p)
    integer, intent(inout) :: p
  end
  subroutine data_default(p)
    integer :: p
  end
  subroutine data_value(p)
    integer, value :: p
  end
end module

module component_interfaces
  use propagation, only: node, callback
  implicit none
  interface
    subroutine declared_in(p)
      integer, pointer, intent(in) :: p
    end
    subroutine declared_default(p)
      integer, pointer :: p
    end
    subroutine declared_protected_in(p)
      integer, pointer, protected_target, intent(in) :: p
    end
    subroutine declared_protected_default(p)
      integer, pointer, protected_target :: p
    end
    subroutine declared_procedure_in(p)
      import callback
      procedure(callback), pointer, intent(in) :: p
    end
    subroutine declared_procedure_default(p)
      import callback
      procedure(callback), pointer :: p
    end
  end interface
end module

subroutine separate_module_caller(root, ordinary)
  use component_interfaces
  implicit none
  type(node), pointer, protected_target :: root
  type(node) :: ordinary
  !ERROR: PROTECTED_TARGET pointer 'root' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call declared_in(root%separate)
  !ERROR: PROTECTED_TARGET pointer 'root' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call declared_default(root%separate)
  call declared_protected_in(root%separate)
  call declared_protected_default(root%separate)
  !ERROR: PROTECTED_TARGET pointer 'root' or its subobject may not be associated with procedure pointer dummy argument 'p='
  call declared_procedure_in(root%proc)
  !ERROR: PROTECTED_TARGET pointer 'root' or its subobject may not be associated with procedure pointer dummy argument 'p='
  call declared_procedure_default(root%proc)
  call declared_in(ordinary%separate)
  call declared_default(ordinary%separate)
  call declared_procedure_in(ordinary%proc)
  call declared_procedure_default(ordinary%proc)
end

subroutine statement_function(p)
  implicit none
  integer, pointer, protected_target :: p
  integer :: f, a, value
  f(a) = a
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
  value = f(p)
end

subroutine implicit_sub(a)
  integer :: a
end

module protected_storage
  implicit none
  type plain
    integer, pointer :: q
  end type
  type protected
    integer, pointer, protected_target :: q
  end type
  integer, protected, target :: x = 1
  integer, pointer, protected :: ordinary_module_pointer
  type(plain), protected, target :: object
contains
  subroutine owner
    integer, pointer, protected_target :: p
    type(protected) :: v
    p => x
    v = protected(x)
    x = 2
  end
  subroutine pointer_in(p)
    integer, pointer, protected_target, intent(in) :: p
  end
end module

subroutine protected_target_conflict
  use protected_storage
  use propagation, only: from_argument
  implicit none
  integer, pointer, protected_target :: p
  !ERROR: Nonpointer USE-associated PROTECTED object 'x' may not be a pointer target, even for a PROTECTED_TARGET pointer
  integer, pointer, protected_target :: initial => x
  type(protected) :: v
  !ERROR: Nonpointer USE-associated PROTECTED object 'x' may not be a pointer target, even for a PROTECTED_TARGET pointer
  p => x
  !ERROR: Nonpointer USE-associated PROTECTED object 'x' may not be a pointer target, even for a PROTECTED_TARGET pointer
  v = protected(x)
  !ERROR: Nonpointer USE-associated PROTECTED object 'x' may not be a pointer target, even for a PROTECTED_TARGET pointer
  call pointer_in(x)
  p => ordinary_module_pointer
  p => object%q
  p => from_argument(x) ! Reading x as an INTENT(IN) argument is permitted.
end

! Same-kind conversions produce values, not protected pointer actuals.

module protected_target_values
  implicit none
  type node
    real :: value
  end type
  integer, target :: target_integer = 1
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => target_integer
  end function

  subroutine scalars(p, r, c, d, ordinary)
    integer, pointer, protected_target :: p
    real, pointer, protected_target :: r
    complex, pointer, protected_target :: c
    double precision, pointer, protected_target :: d
    integer, pointer :: ordinary
    external implicit_integer
    call read_integer(int(p))
    call read_real(real(r))
    call read_complex(cmplx(c))
    call read_double(dble(d))
    call implicit_integer(int(p))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
    call implicit_integer(p)
    call integer_in(p)
    !ERROR: PROTECTED_TARGET pointer 'r' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer(get())
    !ERROR: PROTECTED_TARGET pointer 'r' or its subobject requires an explicit interface
    call implicit_integer(get())
    call read_integer(int(get()))
    call implicit_integer(int(get()))
    call read_integer((int(p)))
    call read_integer((p))
    call read_integer(+p)
    call read_integer(p + 0)
    call read_integer(ordinary)
    call read_integer(int(ordinary))
    call implicit_integer(ordinary)
  end subroutine

  subroutine subobjects(a, root)
    integer, pointer, protected_target :: a(:)
    type(node), pointer, protected_target :: root
    external implicit_integer, implicit_real
    call read_integer(int(a(1)))
    call read_real(real(root%value))
    call implicit_integer(int(a(1)))
    call implicit_real(real(root%value))
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer(a(1))
    !ERROR: PROTECTED_TARGET pointer 'root' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_real(root%value)
    !ERROR: PROTECTED_TARGET pointer 'a' or its subobject requires an explicit interface
    call implicit_integer(a(1))
    !ERROR: PROTECTED_TARGET pointer 'root' or its subobject requires an explicit interface
    call implicit_real(root%value)
  end subroutine

  subroutine associates(p)
    integer, pointer, protected_target :: p
    external implicit_integer
    associate(i => int(p))
      call read_integer(i)
      call implicit_integer(i)
    end associate
    associate(a => p)
      !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
      call read_integer(a)
      !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
      call implicit_integer(a)
      call integer_in(a)
    end associate
    associate(a => int(get()))
      call read_integer(a)
      call implicit_integer(a)
    end associate
    associate(a => get())
      !ERROR: PROTECTED_TARGET pointer 'r' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
      call read_integer(a)
      !ERROR: PROTECTED_TARGET pointer 'r' or its subobject requires an explicit interface
      call implicit_integer(a)
    end associate
  end subroutine

  subroutine conditionals(p, ordinary, choose)
    integer, pointer, protected_target :: p
    integer, pointer :: ordinary
    logical :: choose
    call read_integer((choose ? int(p) : 1))
    call read_integer((choose ? 1 : int(p)))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer((choose ? p : 1))
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer((choose ? 1 : p))
    !ERROR: PROTECTED_TARGET pointer 'r' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call read_integer((choose ? get() : 1))
    call read_integer((choose ? int(get()) : 1))
    call read_integer((choose ? (int(p)) : 1))
    call read_integer((choose ? (p) : 1))
    call read_integer((choose ? +p : 1))
    call read_integer((choose ? p + 0 : 1))
    call read_integer((choose ? int(ordinary) : 1))
    call read_integer((choose ? ordinary : 1))
    call integer_in((choose ? p : 1))
  end subroutine

  subroutine nonvariables(p)
    integer, pointer, protected_target :: p, keep
    integer, pointer :: ordinary
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: '(p)' is not a variable or pointer
    call integer_out(int(p))
    !ERROR: Target associated with pointer 'keep' must be a designator or a call to a pointer-valued function
    keep => int(p)
    !ERROR: Target associated with pointer 'ordinary' must be a designator or a call to a pointer-valued function
    ordinary => int(p)
  end subroutine

  subroutine read_integer(x)
    integer :: x
    print *, x
  end subroutine
  subroutine read_real(x)
    real :: x
    print *, x
  end subroutine
  subroutine read_complex(x)
    complex :: x
    print *, x
  end subroutine
  subroutine read_double(x)
    double precision :: x
    print *, x
  end subroutine
  subroutine integer_in(x)
    integer, intent(in) :: x
    print *, x
  end subroutine
  subroutine integer_out(x)
    integer, intent(out) :: x
    x = 1
  end subroutine
end module

subroutine implicit_integer(x)
  integer :: x
  print *, x
end subroutine
subroutine implicit_real(x)
  real :: x
  print *, x
end subroutine

module protected_target_characteristics
  implicit none
  type base
  contains
    procedure, nopass :: np => ordinary_np
    procedure :: pp => ordinary_pp
    procedure, nopass :: fr => ordinary_fr
  end type
  type, extends(base) :: extension
  contains
    procedure, nopass :: np => protected_np
    procedure :: pp => protected_pp
    procedure, nopass :: fr => protected_fr
  end type
  type holder
    procedure(ordinary_np), pointer, nopass :: callback => protected_np
  end type
  abstract interface
    subroutine default_data(a)
      integer :: a
    end
  end interface
  interface
    module subroutine separate(p, proc)
      integer, pointer, protected_target, intent(in) :: p
      procedure(ordinary_np) :: proc
    end
    module function separate_result() result(r)
      integer, pointer, protected_target :: r
    end
  end interface
contains
  subroutine ordinary_np(p)
    integer, pointer, intent(in) :: p
  end
  subroutine protected_np(p)
    integer, pointer, protected_target, intent(in) :: p
  end
  subroutine optional_np(p)
    integer, pointer, protected_target, intent(in), optional :: p
  end
  subroutine ordinary_pp(this, p)
    class(base), intent(in) :: this
    integer, pointer, intent(in) :: p
  end
  subroutine protected_pp(this, p)
    class(extension), intent(in) :: this
    integer, pointer, protected_target, intent(in) :: p
  end
  function ordinary_fr() result(r)
    integer, pointer :: r
    nullify(r)
  end
  function protected_fr() result(r)
    integer, pointer, protected_target :: r
    nullify(r)
  end
  function entry_result() result(r)
    integer, pointer, protected_target :: r
    integer, pointer :: s
    entry ordinary_entry() result(s)
    nullify(s)
  end
  subroutine interface_calls(p, dp)
    integer, pointer, protected_target :: p
    procedure(default_data) :: dp
    procedure(ordinary_np), pointer :: op
    op => protected_np
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call op(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'a='
    call dp(p)
    call separate(p, protected_np)
    !ERROR: Procedure pointer 'op' associated with incompatible procedure designator 'optional_np': incompatible dummy argument #1: incompatible dummy data object attributes: Optional
    op => optional_np
  end
end module

submodule (protected_target_characteristics) protected_target_bodies
contains
  module subroutine separate(p, proc)
    integer, pointer, intent(in) :: p
    procedure(protected_np) :: proc
  end
  module function separate_result() result(r)
    integer, pointer :: r
    nullify(r)
  end
end submodule

module protected_target_generic
  use protected_target_characteristics, only: ordinary_np, protected_np
  !ERROR: Generic 'ambiguous' may not have specific procedures 'ordinary_np' and 'protected_np' as their interfaces are not distinguishable
  interface ambiguous
    procedure ordinary_np, protected_np
  end interface
end module

subroutine protected_target_external(p)
  integer, pointer, protected_target, intent(in) :: p
end

subroutine protected_target_local_interface
  interface
    subroutine protected_target_external(p)
      integer, pointer, intent(in) :: p
    end
  end interface
  integer, pointer :: p
  call protected_target_external(p)
end
