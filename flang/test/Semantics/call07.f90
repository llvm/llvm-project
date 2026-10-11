! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test 15.5.2.7 constraints and restrictions for POINTER dummy arguments.

module m
  real :: coarray(10)[*]
 contains

  subroutine s01(p)
    real, pointer, contiguous, intent(in) :: p(:)
  end subroutine
  subroutine s02(p)
    real, pointer :: p(:)
  end subroutine
  subroutine s03(p)
    real, pointer, intent(in) :: p(:)
  end subroutine
  subroutine s04(p)
    real, pointer :: p
  end subroutine

  subroutine test
    !PORTABILITY: CONTIGUOUS entity 'a01' should be an array pointer, assumed-shape, or assumed-rank [-Wredundant-contiguous]
    real, pointer, contiguous :: a01 ! C830
    real, pointer :: a02(:)
    real, target :: a03(10)
    real :: a04(10) ! not TARGET
    !PORTABILITY: CONTIGUOUS entity 'scalar' should be an array pointer, assumed-shape, or assumed-rank [-Wredundant-contiguous]
    real, contiguous :: scalar
    call s01(a03) ! ok
    !ERROR: CONTIGUOUS pointer dummy argument may not be associated with non-CONTIGUOUS pointer actual argument
    call s01(a02)
    !WARNING: Target of CONTIGUOUS pointer association is not known to be contiguous [-Wpointer-to-possible-noncontiguous]
    call s01(a02(:))
    !ERROR: CONTIGUOUS pointer may not be associated with a discontiguous target
    call s01(a03(::2))
    call s02(a02) ! ok
    call s03(a03) ! ok
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s02(a03)
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s04(a02(1))
    !ERROR: An array section with a vector subscript may not be a pointer target
    call s03(a03([1,2,4]))
    !ERROR: A coindexed object may not be a pointer target
    call s03(coarray(:)[1])
    !ERROR: Target associated with dummy argument 'p=' must be a designator or a call to a pointer-valued function
    call s03([1.])
    !ERROR: In assignment to object dummy argument 'p=', the target 'a04' is not an object with POINTER or TARGET attributes
    call s03(a04)
  end subroutine
end module

module protected_target_calls
  type node
    integer :: value
    integer, pointer :: link
    type(node), pointer :: next
  end type
  integer, target :: value
  interface generic_rank
    module procedure no_intent_array, elemental_read
  end interface
  interface generic_read
    module procedure read_array, elemental_read
  end interface
  interface generic_ordinary_pointer
    module procedure ordinary_array_pointer
  end interface
  interface generic_protected_pointer
    module procedure readonly_array_pointer
  end interface
  interface generic_scalar
    module procedure no_intent
  end interface
contains
  elemental subroutine elemental_read(x)
    integer, intent(in) :: x
  end subroutine
  subroutine read_scalar(x)
    integer, intent(in) :: x
  end subroutine
  subroutine read_array(x)
    integer, intent(in) :: x(:)
  end subroutine
  subroutine no_intent(x)
    integer :: x
  end subroutine
  subroutine no_intent_array(x)
    integer :: x(:)
  end subroutine
  subroutine write_scalar(x)
    integer, intent(out) :: x
  end subroutine
  subroutine value_scalar(x)
    integer, value :: x
  end subroutine
  subroutine ordinary_pointer(x)
    integer, pointer, intent(in) :: x
  end subroutine
  subroutine ordinary_array_pointer(x)
    integer, pointer, intent(in) :: x(:)
  end subroutine
  subroutine readonly_pointer(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  subroutine readonly_array_pointer(x)
    integer, pointer, protected_target, intent(in) :: x(:)
  end subroutine
  subroutine reassociate(x)
    integer, pointer, protected_target, intent(out) :: x
    nullify(x)
  end subroutine
  subroutine assumed_rank_in(x)
    integer, intent(in) :: x(..)
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => value
  end function
  subroutine test(p, a, object, pin)
    integer, pointer, protected_target :: p, a(:)
    type(node), pointer, protected_target :: object
    integer, pointer, protected_target, intent(in) :: pin
    call read_scalar(p)
    call read_array(a)
    call readonly_pointer(p)
    call readonly_array_pointer(a(1:2))
    call reassociate(p)
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call no_intent(p)
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call no_intent_array(a(1:2))
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call write_scalar(p)
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call value_scalar(p)
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call ordinary_pointer(p)
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call ordinary_array_pointer(a(1:2))
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'pin' is an INTENT(IN) dummy argument
    call reassociate(pin)
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call no_intent(object%value)
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call ordinary_pointer(object%link)
    call readonly_pointer(object%link)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'object' is a PROTECTED_TARGET pointer
    call reassociate(object%link)
    call write_scalar(object%link)
    call ordinary_pointer(object%next%link)
    associate(x => object%value)
      !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
      call no_intent(x)
    end associate
    associate(x => object%next%value)
      call no_intent(x)
    end associate
    call readonly_pointer(get())
    call reassociate(object%next%link)
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call ordinary_pointer(get())
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call no_intent(get())
    associate(x => get())
      !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
      call no_intent(x)
    end associate
    call no_intent(p + 1)
    call no_intent((p))
    call no_intent(get() + 1)
    call no_intent((get()))
    call read_scalar((.true. ? p : value))
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call no_intent((.true. ? p : value))
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call ordinary_pointer((.true. ? p : value))
    call readonly_pointer((.true. ? p : value))
    print *, associated(p), size(a), shape(a), kind(p)
  end subroutine
  subroutine ranks(p)
    integer, pointer, protected_target, intent(in) :: p(..)
    select rank(p)
    rank(0)
      call read_scalar(p)
      call readonly_pointer(p)
      !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
      call no_intent(p)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer
      p = 1
    rank(1)
      call read_array(p)
      !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
      call ordinary_array_pointer(p)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer
      p(1) = 1
    rank default
      call assumed_rank_in(p)
    end select
  end subroutine
  subroutine generic_calls(p, ordinary)
    integer, pointer, protected_target :: p(:)
    integer, pointer :: ordinary(:)
    ! F'2028 C874 is checked after selecting the non-elemental specific.
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call generic_rank(p)
    call generic_rank(ordinary)
    call generic_read(p)
    !ERROR: PROTECTED_TARGET actual argument requires PROTECTED_TARGET for pointer dummy argument 'x='
    call generic_ordinary_pointer(p)
    call generic_ordinary_pointer(ordinary)
    call generic_protected_pointer(p)
    !ERROR: PROTECTED_TARGET actual argument requires INTENT(IN) for nonpointer dummy argument 'x='
    call generic_scalar(p(1))
    call generic_scalar(ordinary(1))
  end subroutine
  subroutine null_inquiries(p, array, ordinary)
    integer, pointer, protected_target :: p, array(:)
    integer, pointer :: ordinary
    procedure(read_scalar), pointer :: callback
    p => null(p)
    p => null(mold=p)
    array => null(array)
    array => null(mold=array)
    ordinary => null(p)
    ordinary => null(mold=ordinary)
    callback => null(callback)
    callback => null(mold=callback)
    call readonly_pointer(null(p))
    call readonly_array_pointer(null(mold=array))
    call ordinary_pointer(null(mold=p))
  end subroutine
end module

module protected_target_protected_actuals
  integer, target, protected :: x = 1
contains
  subroutine read_protected(p)
    integer, pointer, protected_target, intent(in) :: p
  end subroutine
  subroutine read_ordinary(p)
    integer, pointer, intent(in) :: p
  end subroutine
  subroutine owner
    call read_protected(x)
    call read_ordinary(x)
  end subroutine
end module

subroutine protected_target_protected_actual
  use protected_target_protected_actuals
  ! Argument association is not a C864 data-target or initial-data-target.
  call read_protected(x)
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  call read_ordinary(x)
end subroutine

module protected_target_procedure_arguments
  integer, target :: value
contains
  function ordinary_result() result(r)
    integer, pointer :: r
    r => value
  end function
  function protected_result() result(r)
    integer, pointer, protected_target :: r
    r => value
  end function
  subroutine ordinary_dummy(p)
    integer, pointer, intent(in) :: p
  end subroutine
  subroutine protected_dummy(p)
    integer, pointer, protected_target, intent(in) :: p
  end subroutine
  subroutine take_ordinary_result(f)
    procedure(ordinary_result) :: f
  end subroutine
  subroutine take_protected_result(f)
    procedure(protected_result) :: f
  end subroutine
  subroutine take_ordinary_dummy(s)
    procedure(ordinary_dummy) :: s
  end subroutine
  subroutine take_protected_dummy(s)
    procedure(protected_dummy) :: s
  end subroutine
  subroutine take_ordinary_callback(c)
    procedure(take_ordinary_result) :: c
  end subroutine
  subroutine take_protected_callback(c)
    procedure(take_protected_result) :: c
  end subroutine
  subroutine replace_pointer(p)
    integer, pointer, protected_target, intent(inout) :: p
    p => protected_result()
  end subroutine
  subroutine replace_function(f)
    procedure(protected_result), pointer, intent(inout) :: f
    f => protected_result
  end subroutine
  subroutine calls
    ! J3/26-007r1 omits PROTECTED_TARGET from procedure characteristics.
    call take_protected_result(ordinary_result)
    call take_ordinary_dummy(protected_dummy)
    call take_ordinary_result(protected_result)
    call take_protected_dummy(ordinary_dummy)
    call take_ordinary_callback(take_protected_result)
    call take_protected_callback(take_ordinary_result)
  end subroutine
  subroutine returned_associations
    integer, pointer :: p
    procedure(ordinary_result), pointer :: f
    ! No constraint applies: the actual arguments are not PROTECTED_TARGET.
    call replace_pointer(p)
    p = 1
    f => ordinary_result
    call replace_function(f)
    p => f()
    p = 1
  end subroutine
end module

subroutine protected_target_implicit(p)
  integer, pointer, protected_target :: p(:)
  external consume
  !ERROR: PROTECTED_TARGET actual argument requires an explicit interface
  call consume(p)
  !ERROR: PROTECTED_TARGET actual argument requires an explicit interface
  call consume(p(1))
  !ERROR: PROTECTED_TARGET actual argument requires an explicit interface
  call consume(p(1:2))
end subroutine

subroutine protected_target_known_external(p)
  integer, pointer, protected_target :: p
  !ERROR: PROTECTED_TARGET actual argument requires an explicit interface
  call protected_target_external_in(p)
end subroutine

subroutine protected_target_external_in(x)
  integer, intent(in) :: x
end subroutine

subroutine protected_target_external_interfaces(p, ordinary)
  integer, pointer, protected_target :: p
  integer, pointer :: ordinary
  ! These interfaces differ from the definitions below only in PROTECTED_TARGET,
  ! which is not a procedure characteristic.
  interface
    subroutine ordinary_external_dummy(x)
      integer, pointer, protected_target, intent(in) :: x
    end subroutine
    subroutine protected_external_dummy(x)
      integer, pointer, intent(in) :: x
    end subroutine
    function protected_external_result() result(r)
      integer, pointer :: r
    end function
    function ordinary_external_result() result(r)
      integer, pointer, protected_target :: r
    end function
  end interface
  call ordinary_external_dummy(p)
  call protected_external_dummy(ordinary)
  ordinary => protected_external_result()
  p => ordinary_external_result()
end subroutine

subroutine ordinary_external_dummy(x)
  integer, pointer, intent(in) :: x
end subroutine
subroutine protected_external_dummy(x)
  integer, pointer, protected_target, intent(in) :: x
end subroutine
function ordinary_external_result() result(r)
  integer, pointer :: r
  r => null()
end function
function protected_external_result() result(r)
  integer, pointer, protected_target :: r
  r => null()
end function
