! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C872-C874 and procedure characteristics.
! PROTECTED_TARGET is not a procedure characteristic in F2028.

module protected_target_calls
  use iso_c_binding, only: c_ptr, c_f_pointer
  type t
    integer :: n
    integer, pointer :: link
  end type
  integer, target :: x
  interface
    subroutine ignore_pointer(q)
      integer, pointer :: q
      !dir$ ignore_tkr(p) q
    end subroutine
  end interface
contains
  subroutine calls(p, a, obj)
    integer, pointer, protected_target, optional :: p
    integer, pointer, protected_target :: a(:)
    integer, pointer :: w
    type(t), pointer, protected_target :: obj
    type(c_ptr) :: cp
    external implicit
    print *, associated(p), size(a), lbound(a), present(p)
    call read_value(p)
    call read_value(obj%n)
    call read_array(a(1:2))
    call read_pointer(p)
    call out_pointer(p)
    call inout_pointer(p)
    call default_pointer(p)
    ! A value expression no longer designates the protected target.
    call default_value((p))
    call default_value(p + 1)
    call implicit((p))
    !ERROR: PROTECTED_TARGET actual argument 'p' requires an explicit interface
    call implicit(p)
    !ERROR: PROTECTED_TARGET actual argument 'a(1_8:2_8:1_8)' requires an explicit interface
    call implicit(a(1:2))
    !ERROR: Nonpointer dummy argument 'q=' associated with PROTECTED_TARGET actual argument 'p' must have INTENT(IN)
    call default_value(p)
    !ERROR: Actual argument type 'INTEGER(4)' is not compatible with dummy argument type 'REAL(4)'
    !ERROR: Nonpointer dummy argument 'q=' associated with PROTECTED_TARGET actual argument 'p' must have INTENT(IN)
    call default_real(p)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'q=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    call out_value(p)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'q=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    call inout_value(p)
    !ERROR: Nonpointer dummy argument 'q=' associated with PROTECTED_TARGET actual argument 'p' must have INTENT(IN)
    call copied_value(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ordinary_pointer(p)
    !ERROR: PROTECTED_TARGET pointer 'a' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ordinary_array(a(1:2))
    !ERROR: Nonpointer dummy argument 'q=' associated with PROTECTED_TARGET actual argument 'obj%n' must have INTENT(IN)
    call default_value(obj%n)
    !ERROR: Actual argument associated with POINTER dummy argument 'q=' must also be POINTER unless INTENT(IN)
    call ordinary_default_pointer(obj%n)
    ! IGNORE_TKR(P) must not bypass F2028 C872.
    !ERROR: PROTECTED_TARGET pointer 'obj' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ignore_pointer(obj%n)
    call ignore_pointer(x)
    ! Intrinsic pointer dummies are exempt from F2028 C872, not definability checks.
    w => null(p)
    print *, associated(w, p)
    call c_f_pointer(cp, p)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'fptr=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    call c_f_pointer(cp, obj%link)
    !ERROR: The PROTECTED_TARGET result of 'get' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ordinary_pointer(get())
    !ERROR: PROTECTED_TARGET actual argument 'get()' requires an explicit interface
    call implicit(get())
    call read_value(get())
    ! Unlike data-target and constructor uses, a pointer actual aliases obj%link itself.
    !ERROR: PROTECTED_TARGET pointer 'obj' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ordinary_default_pointer(obj%link)
    !ERROR: PROTECTED_TARGET pointer 'obj' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call ordinary_pointer(obj%link)
    call read_pointer(obj%link)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'q=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    call out_pointer(obj%link)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'q=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    call inout_pointer(obj%link)
    call default_value(obj%link)
    ! Associate names do not have PROTECTED_TARGET (F2028 C872-C874).
    associate (alias => obj%n)
      call implicit(alias)
      call default_value(alias)
      call read_value(alias)
      call ordinary_pointer(alias)
      !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'q=' is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
      call inout_value(alias)
    end associate
  end subroutine

  subroutine read_value(q)
    integer, intent(in) :: q
  end subroutine
  subroutine read_array(q)
    integer, intent(in) :: q(:)
  end subroutine
  subroutine default_value(q)
    integer :: q
  end subroutine
  subroutine default_real(q)
    real :: q
  end subroutine
  subroutine out_value(q)
    integer, intent(out) :: q
  end subroutine
  subroutine inout_value(q)
    integer, intent(inout) :: q
  end subroutine
  subroutine copied_value(q)
    integer, value :: q
  end subroutine
  subroutine read_pointer(q)
    integer, pointer, protected_target, intent(in) :: q
  end subroutine
  subroutine out_pointer(q)
    integer, pointer, protected_target, intent(out) :: q
    nullify(q)
  end subroutine
  subroutine inout_pointer(q)
    integer, pointer, protected_target, intent(inout) :: q
    nullify(q)
  end subroutine
  subroutine default_pointer(q)
    integer, pointer, protected_target :: q
  end subroutine
  subroutine ordinary_pointer(q)
    integer, pointer, intent(in) :: q
  end subroutine
  subroutine ordinary_default_pointer(q)
    integer, pointer :: q
    q => x
  end subroutine
  subroutine ordinary_array(q)
    integer, pointer, intent(in) :: q(:)
  end subroutine
  function get() result(p)
    integer, pointer, protected_target :: p
    p => x
  end function
  function ordinary_get() result(p)
    integer, pointer :: p
    p => x
  end function

  subroutine procedure_pointers
    procedure(read_pointer), pointer :: s
    procedure(ordinary_pointer), pointer :: so
    procedure(get), pointer :: f
    procedure(ordinary_get), pointer :: fo
    s => read_pointer
    f => get
    s => ordinary_pointer
    so => read_pointer
    f => ordinary_get
    fo => get
    call takes_ordinary(read_pointer)
    call takes_protected(ordinary_pointer)
    call takes_protected_result(ordinary_get)
    call takes_ordinary_result(get)
  end subroutine

  subroutine takes_ordinary(proc)
    procedure(ordinary_pointer) :: proc
  end subroutine
  subroutine takes_protected(proc)
    procedure(read_pointer) :: proc
  end subroutine
  subroutine takes_ordinary_result(proc)
    procedure(ordinary_get) :: proc
  end subroutine
  subroutine takes_protected_result(proc)
    procedure(get) :: proc
  end subroutine

  subroutine statement_function(p)
    integer, pointer, protected_target :: p
    integer :: sf, q, n
    sf(q) = q
    !ERROR: PROTECTED_TARGET actual argument 'p' requires an explicit interface
    n = sf(p)
    n = sf((p))
  end subroutine
end module

module protected_target_generics
  type t
    integer :: n
  contains
    procedure :: assign_int
    generic :: assignment(=) => assign_int
  end type
  type u
    integer :: n
  end type
  type box
    integer, pointer, protected_target :: p
  end type
  interface h
    module procedure h_array, h_elemental
  end interface
  interface box
    module procedure make_box
  end interface
  interface assignment(=)
    module procedure assign_u
  end interface
  interface k
    module procedure pointer_arg
  end interface
contains
  subroutine h_array(x)
    integer :: x(:)
    x = 0
  end subroutine
  elemental subroutine h_elemental(x)
    integer, intent(in) :: x
  end subroutine
  function make_box(x) result(r)
    integer :: x
    type(box) :: r
    x = 99
  end function
  subroutine assign_int(this, v)
    class(t), intent(inout) :: this
    integer, intent(in) :: v
    this%n = v
  end subroutine
  subroutine assign_u(x, v)
    type(u), intent(out) :: x
    integer, intent(in) :: v
    x%n = v
  end subroutine
  subroutine pointer_arg(q)
    integer, pointer, intent(in) :: q
  end subroutine
  subroutine calls(a, p, q, r, ordinary)
    integer, pointer, protected_target :: a(:), p
    type(t), pointer, protected_target :: q
    type(u), pointer, protected_target :: r
    integer, pointer :: ordinary(:)
    type(box) :: b
    ! Resolve the nonelemental specific before checking F2028 C874.
    !ERROR: Nonpointer dummy argument 'x=' associated with PROTECTED_TARGET actual argument 'a' must have INTENT(IN)
    call h(a)
    call h(ordinary)
    ! Resolve the generic function, not the same-name structure constructor.
    !ERROR: Nonpointer dummy argument 'x=' associated with PROTECTED_TARGET actual argument 'p' must have INTENT(IN)
    b = box(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or a subobject may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call k(p)
    ! Defined assignments resolve before checking target definability.
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'this=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'q' is not definable
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'q' is not definable
    q = 3
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'r' is not definable
    r = 4
  end subroutine
end module

module protected_target_separate
  use protected_target_calls, only: read_pointer, ordinary_pointer, get, ordinary_get
  interface
    module subroutine sep(q)
      integer, pointer, protected_target, intent(in) :: q
    end subroutine
    module subroutine sep_protected(q)
      integer, pointer, intent(in) :: q
    end subroutine
    module function result_plain() result(p)
      integer, pointer, protected_target :: p
    end function
    module function result_protected() result(p)
      integer, pointer :: p
    end function
    module subroutine procedure_args(s1, s2, f1, f2)
      procedure(read_pointer) :: s1
      procedure(ordinary_pointer) :: s2
      procedure(get) :: f1
      procedure(ordinary_get) :: f2
    end subroutine
  end interface
end module
submodule(protected_target_separate) protected_target_separate_impl
contains
  module subroutine sep(q)
    integer, pointer, intent(in) :: q
  end subroutine
  module subroutine sep_protected(q)
    integer, pointer, protected_target, intent(in) :: q
  end subroutine
  module function result_plain() result(p)
    integer, pointer :: p
    nullify(p)
  end function
  module function result_protected() result(p)
    integer, pointer, protected_target :: p
    nullify(p)
  end function
  module subroutine procedure_args(s1, s2, f1, f2)
    procedure(ordinary_pointer) :: s1
    procedure(read_pointer) :: s2
    procedure(ordinary_get) :: f1
    procedure(get) :: f2
  end subroutine
end submodule

module protected_target_overrides
  use protected_target_calls, only: read_pointer, ordinary_pointer, get, ordinary_get
  type base
  contains
    procedure, nopass :: dummy_protected => ordinary_pointer
    procedure, nopass :: dummy_plain => read_pointer
    procedure, nopass :: result_plain => get
    procedure, nopass :: result_protected => ordinary_get
  end type
  type, extends(base) :: extended
  contains
    procedure, nopass :: dummy_protected => read_pointer
    procedure, nopass :: dummy_plain => ordinary_pointer
    procedure, nopass :: result_plain => ordinary_get
    procedure, nopass :: result_protected => get
  end type
end module
