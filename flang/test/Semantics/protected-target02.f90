! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable -Wmismatching-dummy-procedure

module targets
  integer, target, protected :: x
  type t
    integer :: value
    integer, pointer :: link
  contains
    procedure :: passed_unspecified
  end type
  type(t), target, protected :: obj
  type writable
    integer, pointer :: c
  end type
  type readonly
    integer, pointer, protected_target :: c
  end type
  interface generic_value
    module procedure value_inout
  end interface
  interface generic_pointer
    module procedure pointer_in
  end interface
  interface assignment(=)
    module procedure set_value
  end interface
contains
  subroutine owner
    integer, pointer, protected_target :: p
    p => x
    p => obj%value
    print *, p
  end
  function get() result(p)
    integer, pointer, protected_target :: p
    p => x
  end
  subroutine pointer_in(p)
    integer, pointer, intent(in) :: p
  end
  subroutine pointer_unspecified(p)
    integer, pointer :: p
    nullify(p)
  end
  subroutine pointer_out(p)
    integer, pointer, intent(out) :: p
    nullify(p)
  end
  subroutine pointer_inout(p)
    integer, pointer, intent(inout) :: p
    nullify(p)
  end
  subroutine readonly_in(p)
    integer, pointer, protected_target, intent(in) :: p
  end
  subroutine readonly_unspecified(p)
    integer, pointer, protected_target :: p
    nullify(p)
  end
  subroutine readonly_out(p)
    integer, pointer, protected_target, intent(out) :: p
    nullify(p)
  end
  subroutine readonly_inout(p)
    integer, pointer, protected_target, intent(inout) :: p
    nullify(p)
  end
  subroutine value_in(a)
    integer, intent(in) :: a
  end
  subroutine value_out(a)
    integer, intent(out) :: a
  end
  subroutine value_inout(a)
    integer, intent(inout) :: a
  end
  subroutine value_unspecified(a)
    integer :: a
  end
  subroutine value_only(a)
    integer, value :: a
  end
  subroutine passed_unspecified(self)
    class(t) :: self
  end
  elemental subroutine elemental_inout(a)
    integer, intent(inout) :: a
  end
  subroutine set_value(a, b)
    type(t), intent(out) :: a
    integer, intent(in) :: b
    a%value = b
  end
end

subroutine associations
  use targets
  integer, target :: local, array(2)
  integer, pointer :: w, section(:)
  integer, pointer, protected_target :: p, q, pa(:)
  type(t), pointer, protected_target :: pt
  type(writable) :: a
  type(readonly) :: b
  ! Follow F2028 8.5.16 NOTE 1 despite its conflict with C864.
  p => x
  p => obj%value
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  w => x
  w => local
  p => w
  q => p
  pa => array
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
  w => p
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject may not be associated with pointer 'section' without PROTECTED_TARGET
  section => pa(1:2)
  pt => obj
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
  w => pt%value
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'c' without PROTECTED_TARGET
  a = writable(p)
  b = readonly(p)
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject may not be associated with pointer 'c' without PROTECTED_TARGET
  a = writable(pa(1))
  b = readonly(pa(1))
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
  w => get()
  p => get()
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'c' without PROTECTED_TARGET
  a = writable(get())
  b = readonly(get())
  associate(alias => pa(1:2))
    !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject may not be associated with pointer 'section' without PROTECTED_TARGET
    section => alias
  end associate
  nullify(p, q, pa, pt)
end

subroutine initial_data_targets
  use targets, only: x
  ! Follow F2028 8.5.16 NOTE 1 for initial-data-targets too.
  integer, pointer, protected_target, save :: p => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  integer, pointer, save :: ordinary => x
end

subroutine calls
  use targets
  integer, pointer, protected_target :: p, pa(:)
  type(t), pointer, protected_target :: pt
  external implicit_interface
  call readonly_in(p)
  call readonly_out(p)
  call readonly_inout(p)
  call value_in(p)
  call value_in(pa(1))
  call value_in(pt%value)
  call value_in(get())
  call readonly_in(get())
  call readonly_in(pa(1))
  call readonly_in(x)
  call value_out(pt%link)
  call value_unspecified((p))
  call value_unspecified(p + 1)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_in(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_in(get())
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_in(pt%value)
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_in(pa(1))
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_out(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_inout(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_unspecified(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_only(p)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject requires INTENT(IN) for nonpointer dummy argument 'self='
  call pt%passed_unspecified()
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call elemental_inout(pa)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call generic_value(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call generic_pointer(p)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable
  pt = 3
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_unspecified(pa(1))
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
  call value_unspecified(pt%value)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
  call implicit_interface(p)
  !ERROR: PROTECTED_TARGET pointer 'pa' or its subobject requires an explicit interface
  call implicit_interface(pa(1:2))
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject requires an explicit interface
  call implicit_interface(pt%value)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
  call implicit_interface(get())
  associate(alias => pt%value)
    call value_in(alias)
    !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
    call value_unspecified(alias)
  end associate
  associate(alias => get())
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    alias = 1
    call value_in(alias)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires INTENT(IN) for nonpointer dummy argument 'a='
    call value_unspecified(alias)
  end associate
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  get() = 1
end

subroutine pointer_component_actuals
  use targets
  type(t), pointer, protected_target :: pt
  integer, pointer :: w
  type(t) :: value
  w => pt%link
  value = t(pt%link, w)
  call value_in(pt%link)
  call value_out(pt%link)
  pt%link = 1
  call readonly_in(pt%link)
  call readonly_unspecified(pt%link)
  call pointer_in(value%link)
  call pointer_unspecified(value%link)
  call pointer_out(value%link)
  call pointer_inout(value%link)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_in(pt%link)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_unspecified(pt%link)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_out(pt%link)
  !ERROR: PROTECTED_TARGET pointer 'pt' or its subobject may not be associated with dummy argument 'p=' without PROTECTED_TARGET
  call pointer_inout(pt%link)
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'p=' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable
  call readonly_out(pt%link)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable
  call readonly_inout(pt%link)
end

subroutine intrinsic_arguments
  real, pointer, protected_target :: r, a(:)
  integer, pointer, protected_target :: i, ia(:)
  character(:), pointer, protected_target :: c
  real, pointer :: ordinary
  integer :: ordinary_integer
  r => null(r)
  r => null(mold=r)
  a => null(a)
  a => null(mold=a)
  ordinary => null(r)
  ordinary => null(ordinary)
  !ERROR: PROTECTED_TARGET pointer 'r' or its subobject requires INTENT(IN) for nonpointer dummy argument 'harvest='
  call random_number(r)
  !ERROR: PROTECTED_TARGET pointer 'r' or its subobject requires INTENT(IN) for nonpointer dummy argument 'harvest='
  call random_number(harvest=r)
  !ERROR: PROTECTED_TARGET pointer 'a' or its subobject requires INTENT(IN) for nonpointer dummy argument 'harvest='
  call random_number(a)
  !ERROR: PROTECTED_TARGET pointer 'c' or its subobject requires INTENT(IN) for nonpointer dummy argument 'command='
  call get_command(command=c)
  !ERROR: PROTECTED_TARGET pointer 'i' or its subobject requires INTENT(IN) for nonpointer dummy argument 'count='
  call system_clock(count=i)
  !ERROR: PROTECTED_TARGET pointer 'ia' or its subobject requires INTENT(IN) for nonpointer dummy argument 'get='
  call random_seed(get=ia)
  call random_seed(put=ia)
  !ERROR: PROTECTED_TARGET pointer 'i' or its subobject requires INTENT(IN) for nonpointer dummy argument 'to='
  call mvbits(0, 0, 1, i, 0)
  call mvbits(i, 0, 1, ordinary_integer, 0)
end

subroutine c_interoperability
  use iso_c_binding, only: c_f_pointer, c_null_ptr
  integer, pointer :: ordinary
  integer, pointer, protected_target :: p
  call c_f_pointer(c_null_ptr, ordinary)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'fptr=' without PROTECTED_TARGET
  call c_f_pointer(c_null_ptr, p)
end

subroutine defined_assignment_control(y)
  use targets
  type(t), intent(in) :: y
  type(t) :: ordinary
  ordinary = 3
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
  !BECAUSE: 'y' is an INTENT(IN) dummy argument
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'y' is an INTENT(IN) dummy argument
  y = 3
end

subroutine characteristic_matching
  use targets
  procedure(readonly_in), pointer :: proc
  procedure(get), pointer :: func
  interface
    function ordinary_result() result(p)
      integer, pointer :: p
    end
  end interface
  procedure(pointer_in), pointer :: ordinary_proc
  procedure(ordinary_result), pointer :: ordinary_func
  proc => readonly_in
  proc => pointer_in
  func => get
  func => ordinary_result
  ordinary_proc => readonly_in
  ordinary_func => get
  call take_readonly(pointer_in)
  call take_ordinary(readonly_in)
  call take_readonly_function(ordinary_result)
  call take_ordinary_function(get)
contains
  subroutine take_readonly(sub)
    procedure(readonly_in) :: sub
  end
  subroutine take_ordinary(sub)
    procedure(pointer_in) :: sub
  end
  subroutine take_readonly_function(f)
    procedure(get) :: f
  end
  subroutine take_ordinary_function(f)
    procedure(ordinary_result) :: f
  end
end

module separate_characteristics
  interface
    module subroutine ordinary_body(p)
      integer, pointer, protected_target, intent(in) :: p
    end
    module subroutine readonly_body(p)
      integer, pointer, intent(in) :: p
    end
    module function ordinary_body_result() result(p)
      integer, pointer, protected_target :: p
    end
    module function readonly_body_result() result(p)
      integer, pointer :: p
    end
    module subroutine dummy_procedures(readonly, ordinary, readonly_result, ordinary_result)
      interface
        subroutine readonly(p)
          integer, pointer, protected_target, intent(in) :: p
        end
        subroutine ordinary(p)
          integer, pointer, intent(in) :: p
        end
        function readonly_result() result(p)
          integer, pointer, protected_target :: p
        end
        function ordinary_result() result(p)
          integer, pointer :: p
        end
      end interface
    end
    module subroutine optional_mismatch(p)
      integer, pointer, optional, intent(in) :: p
    end
  end interface
end

submodule(separate_characteristics) separate_bodies
contains
  module subroutine ordinary_body(p)
    integer, pointer, intent(in) :: p
  end
  module subroutine readonly_body(p)
    integer, pointer, protected_target, intent(in) :: p
  end
  module function ordinary_body_result() result(p)
    integer, pointer :: p
    nullify(p)
  end
  module function readonly_body_result() result(p)
    integer, pointer, protected_target :: p
    nullify(p)
  end
  module subroutine dummy_procedures(readonly, ordinary, readonly_result, ordinary_result)
    interface
      subroutine readonly(p)
        integer, pointer, intent(in) :: p
      end
      subroutine ordinary(p)
        integer, pointer, protected_target, intent(in) :: p
      end
      function readonly_result() result(p)
        integer, pointer :: p
      end
      function ordinary_result() result(p)
        integer, pointer, protected_target :: p
      end
    end interface
  end
  module subroutine optional_mismatch(p)
    !ERROR: Dummy argument 'p' does not have the OPTIONAL attribute; the corresponding argument in the interface body does
    integer, pointer, protected_target, intent(in) :: p
  end
end

module overriding_characteristics
  type base
  contains
    procedure, nopass :: read => readonly
    procedure, nopass :: write => ordinary
    procedure, nopass :: read_result => readonly_result
    procedure, nopass :: write_result => ordinary_result
  end type
  type, extends(base) :: extended
  contains
    procedure, nopass :: read => ordinary
    procedure, nopass :: write => readonly
    procedure, nopass :: read_result => ordinary_result
    procedure, nopass :: write_result => readonly_result
  end type
contains
  subroutine readonly(p)
    integer, pointer, protected_target, intent(in) :: p
  end
  subroutine ordinary(p)
    integer, pointer, intent(in) :: p
  end
  function readonly_result() result(p)
    integer, pointer, protected_target :: p
    nullify(p)
  end
  function ordinary_result() result(p)
    integer, pointer :: p
    nullify(p)
  end
end
