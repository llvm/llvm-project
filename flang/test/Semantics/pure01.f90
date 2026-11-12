! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that an impure operator can't be called
! from a pure context. tests both type bound and
! non type bound operators.
module m
  type t
   contains
    procedure :: binding => func
    generic :: operator(.not.) => binding
  end type
 contains
  impure integer function func(x)
    class(t), intent(in) :: x
    func = 0
  end
  pure integer function pure_id(n)
    integer, intent(in) :: n
    pure_id = n
  end
  pure integer function test()
    ! A direct call to a pure procedure is allowed.
    test = pure_id(1)
    !ERROR: Procedure 'func' referenced in pure subprogram 'test' must be pure too
    test = .not. t()
  end
end

! F2023 C15105: a procedure referenced from a pure subprogram must be pure,
! including references through defined operators (binary, unary, named,
! and relational). The operator's generic is declared where its interface
! block appears; the purity check must be anchored at the use site instead,
! so each reference is diagnosed independently and the message points at
! the call, not at the (possibly distant, possibly single) declaration.

module pure_op_mod
  implicit none
  type, public :: vec_type
    real :: x = 0.0, y = 0.0
  end type vec_type

  interface operator(+)
    procedure impure_add ! impure
  end interface operator(+)

  interface operator(*)
    procedure impure_scale ! impure
  end interface operator(*)

  interface operator(-)
    procedure pure_sub ! binary, pure
    procedure impure_neg ! unary, impure
  end interface operator(-)

  interface operator(==)
    procedure impure_eq ! impure
  end interface operator(==)

  interface operator(.dot.)
    procedure impure_dot ! impure
  end interface operator(.dot.)

  ! One generic, two specifics with different purity: only the specific
  ! that overload resolution actually picks may be flagged.
  interface operator(.scale.)
    procedure pure_scale_int ! pure, (vec_type, integer)
    procedure impure_scale_real ! impure, (vec_type, real)
  end interface operator(.scale.)

contains
  function impure_add(a, b) result(r)
    type(vec_type), intent(in) :: a, b
    type(vec_type) :: r
    r%x = a%x + b%x
    r%y = a%y + b%y
  end function impure_add

  function impure_scale(s, v) result(r)
    real, intent(in) :: s
    type(vec_type), intent(in) :: v
    type(vec_type) :: r
    r%x = s * v%x
    r%y = s * v%y
  end function impure_scale

  pure function pure_sub(a, b) result(r)
    type(vec_type), intent(in) :: a, b
    type(vec_type) :: r
    r%x = a%x - b%x
    r%y = a%y - b%y
  end function pure_sub

  function impure_neg(a) result(r)
    type(vec_type), intent(in) :: a
    type(vec_type) :: r
    r%x = -a%x
    r%y = -a%y
  end function impure_neg

  function impure_eq(a, b) result(same)
    type(vec_type), intent(in) :: a, b
    logical :: same
    same = a%x == b%x .and. a%y == b%y
  end function impure_eq

  function impure_dot(a, b) result(d)
    type(vec_type), intent(in) :: a, b
    real :: d
    d = a%x * b%x + a%y * b%y
  end function impure_dot

  pure function pure_scale_int(v, n) result(r)
    type(vec_type), intent(in) :: v
    integer, intent(in) :: n
    type(vec_type) :: r
    r%x = v%x * n
    r%y = v%y * n
  end function pure_scale_int

  function impure_scale_real(v, s) result(r)
    type(vec_type), intent(in) :: v
    real, intent(in) :: s
    type(vec_type) :: r
    r%x = v%x * s
    r%y = v%y * s
  end function impure_scale_real

  ! Simple, called from the PURE, ELEMENTAL and SIMPLE cases below so that the
  ! only impurity they can trip over is the impure_add defined operator.
  simple function pure_half(a) result(r)
    type(vec_type), intent(in) :: a
    type(vec_type) :: r
    r%x = a%x * 0.5
    r%y = a%y * 0.5
  end function pure_half

  ! ELEMENTAL is implicitly PURE unless declared IMPURE (F2023 15.7).
  elemental function elemental_caller(p, q) result(r)
    type(vec_type), intent(in) :: p, q
    type(vec_type) :: r
    ! A direct call to a pure procedure is allowed.
    r = pure_half(p)
    !ERROR: Procedure 'impure_add' (as OPERATOR(+)) referenced in pure subprogram 'elemental_caller' must be pure too
    r = pure_half(p + q)
  end function elemental_caller

  ! SIMPLE implies PURE (F2023 15.8), and is stricter.
  simple function simple_caller(p, q) result(r)
    type(vec_type), intent(in) :: p, q
    type(vec_type) :: r
    ! A direct call to a pure procedure is allowed.
    r = pure_half(p)
    !ERROR: Procedure 'impure_add' (as OPERATOR(+)) referenced in pure subprogram 'simple_caller' must be pure too
    r = pure_half(p + q)
  end function simple_caller
end module pure_op_mod

function local_impure_add(a, b) result(z)
  use pure_op_mod, only: vec_type
  type(vec_type), intent(in) :: a, b
  type(vec_type) :: z
  z%x = a%x + b%x
  z%y = a%y + b%y
end function local_impure_add

program main
  use pure_op_mod
  implicit none
  type(vec_type) :: a, b, m
  real :: d
  logical :: l
  a = vec_type(1.0, 2.0)
  b = vec_type(3.0, 4.0)
  ! Impure defined operators are fine outside a pure subprogram.
  m = 0.5 * (a + b)
  m = a - b
  m = -a
  l = a == b
  d = a .dot. b
  m = a .scale. 2
  m = a .scale. 2.0
  m = pure_sub(a, b)
  m = pure_scale_int(a, 2)
  m = pure_half(a)
  m = midpoint(a, b)
contains
  pure function midpoint(p, q) result(r)
    type(vec_type), intent(in) :: p, q
    type(vec_type) :: r
    real :: d
    logical :: eqv
    interface operator(.local.)
      function local_impure_add(a, b) result(z)
        import :: vec_type
        type(vec_type), intent(in) :: a, b
        type(vec_type) :: z
      end function local_impure_add
    end interface operator(.local.)
    ! Direct calls to pure procedures are allowed.
    r = pure_sub(p, q)
    r = pure_scale_int(p, 2)
    r = pure_half(p)
    ! A pure binary defined operator may be referenced.
    r = p - q
    ! Overload resolution picks the pure specific here: no error.
    r = p .scale. 2
    !ERROR: Procedure 'impure_add' (as OPERATOR(+)) referenced in pure subprogram 'midpoint' must be pure too
    !ERROR: Procedure 'impure_scale' (as OPERATOR(*)) referenced in pure subprogram 'midpoint' must be pure too
    r = 0.5 * (p + q)
    !ERROR: Procedure 'impure_neg' (as OPERATOR(-)) referenced in pure subprogram 'midpoint' must be pure too
    r = -p
    !ERROR: Procedure 'impure_eq' (as OPERATOR(==)) referenced in pure subprogram 'midpoint' must be pure too
    eqv = p == q
    !ERROR: Procedure 'impure_dot' (as OPERATOR(.dot.)) referenced in pure subprogram 'midpoint' must be pure too
    d = p .dot. q
    !ERROR: Procedure 'impure_scale_real' (as OPERATOR(.scale.)) referenced in pure subprogram 'midpoint' must be pure too
    r = p .scale. 2.0
    !ERROR: Procedure 'local_impure_add' (as OPERATOR(.local.)) referenced in pure subprogram 'midpoint' must be pure too
    r = p .local. q
    !ERROR: Procedure 'local_impure_add' (as OPERATOR(.local.)) referenced in pure subprogram 'midpoint' must be pure too
    r = q .local. p
  end function midpoint
end program main
