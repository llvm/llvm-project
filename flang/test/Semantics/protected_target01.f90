! RUN: %python %S/test_errors.py %s %flang_fc1
! Fortran 2028 PROTECTED_TARGET constraint checks (C781, C8109, C866-C874)

module m_pt_constraints
  ! C781: Component with PROTECTED_TARGET must also have POINTER
  type t_comp
    !ERROR: PROTECTED_TARGET component 'c' must have the POINTER attribute
    integer, protected_target :: c
  end type

  ! C8109, 8.5.16p1: Entity must be a data pointer
  !ERROR: PROTECTED_TARGET entity 'bad_var' must be a data pointer
  integer, protected_target :: bad_var
  !ERROR: PROTECTED_TARGET entity 'bad_alloc' must be a data pointer
  integer, allocatable, protected_target :: bad_alloc
  !ERROR: PROTECTED_TARGET entity 'bad_param' must be a data pointer
  integer, parameter, protected_target :: bad_param = 1
  !ERROR: PROTECTED_TARGET entity 'bad_target' must be a data pointer
  integer, target, protected_target :: bad_target

  !ERROR: PROTECTED_TARGET entity 'bad_stmt_var' must be a data pointer
  integer :: bad_stmt_var
  protected_target :: bad_stmt_var

  abstract interface
    subroutine sub_iface(); end subroutine
  end interface
  !ERROR: PROTECTED_TARGET entity 'bad_proc_p' must be a data pointer
  procedure(sub_iface), pointer :: bad_proc_p
  protected_target :: bad_proc_p

  ! C867: COMMON block restriction
  !ERROR: A pointer 'common_p' with the PROTECTED_TARGET attribute may not be in a COMMON block
  integer, pointer, protected_target :: common_p
  common /cblock/ common_p

  type sub_t
    integer :: a
    integer, pointer :: pc
    integer, pointer, protected_target :: ptc2 => null()
    integer, allocatable :: al(:)
  end type

  type container_t
    type(sub_t), pointer, protected_target :: ptc
  end type

  type int_ptr_comp_t
    integer, pointer, protected_target :: ptc
  end type

  type poly_comp_t
    class(*), pointer, protected_target :: cp
  end type

contains
  function get_pt() result(r)
    integer, pointer, protected_target :: r
  end function

  function func_pt() result(r)
    integer, pointer, protected_target :: r
    r => null()
  end function

  function func_ord() result(r)
    integer, pointer :: r
    r => null()
  end function

  subroutine test_checks(tgt, tgt_arr)
    integer, target :: tgt
    integer, target :: tgt_arr(2)
    integer, pointer, protected_target :: p_int
    real, pointer, protected_target :: p_real
    integer, pointer, protected_target :: p_arr(:)
    type(sub_t), pointer, protected_target :: p_sub
    type(container_t) :: x
    type(container_t) :: xa(2)
    type(container_t), pointer :: op_sub
    type(int_ptr_comp_t), pointer :: op_int
    type(poly_comp_t), pointer :: op_poly
    type(int_ptr_comp_t) :: x_int
    integer, pointer :: ord_int
    integer, pointer :: ord_arr(:)
    type(sub_t), pointer :: ord_sub
    class(*), pointer, protected_target :: poly_pt
    character(:), pointer, protected_target :: cp
    complex, pointer, protected_target :: zp
    integer, allocatable :: tmp(:)
    procedure(get_pt), pointer :: fp
    integer, pointer :: p_no_colon
    protected_target p_no_colon
    external :: implicit_sub
    namelist /nl_pt/ p_int

    ! C868: DEALLOCATE statement
    !ERROR: A pointer with the PROTECTED_TARGET attribute may not be an allocate-object in a DEALLOCATE statement
    deallocate(p_int)
    !ERROR: A pointer with the PROTECTED_TARGET attribute may not be an allocate-object in a DEALLOCATE statement
    deallocate(x%ptc)
    !ERROR: A pointer with the PROTECTED_TARGET attribute may not be an allocate-object in a DEALLOCATE statement
    deallocate(xa(1)%ptc)

    ! C869: ALLOCATE without SOURCE=
    !ERROR: An ALLOCATE statement for a pointer with the PROTECTED_TARGET attribute must have a SOURCE= clause
    allocate(p_int)
    !ERROR: An ALLOCATE statement for a pointer with the PROTECTED_TARGET attribute must have a SOURCE= clause
    allocate(p_int, mold=tgt)
    !ERROR: An ALLOCATE statement for a pointer with the PROTECTED_TARGET attribute must have a SOURCE= clause
    allocate(x%ptc)

    ! C866 & 8.5.16p2: Definability
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    p_int = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'p_arr' with the PROTECTED_TARGET attribute may not be defined
    p_arr(1:2) = 3
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    p_sub%a = 4
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
    x%ptc%a = 5
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
    x%ptc = sub_t(1, null())
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    p_sub%pc => tgt
    !ERROR: 'pc' may not appear in NULLIFY
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    nullify(p_sub%pc)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    allocate(p_sub%pc)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    allocate(p_sub%al(2))
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    deallocate(p_sub%al)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'from=' is not definable
    !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
    call move_alloc(p_sub%al, tmp)
    !ERROR: Internal file variable 'cp' is not definable
    !BECAUSE: The target of 'cp' with the PROTECTED_TARGET attribute may not be defined
    write(cp, '(a)') 'x'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'cp' with the PROTECTED_TARGET attribute may not be defined
    cp(1:2) = 'ab'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'zp' with the PROTECTED_TARGET attribute may not be defined
    zp%re = 1.0
    !ERROR: Input variable 'p_int' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    read *, p_int
    !ERROR: STAT variable 'p_int' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    allocate(ord_int, stat=p_int)
    !ERROR: 'p_int' may not be used as a DO variable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    do p_int = 1, 2
    end do
    !ERROR: NAMELIST input group must not contain undefinable item 'p_int'
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    read(*, nml=nl_pt)
    !ERROR: IOSTAT variable 'p_int' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    read(*, *, iostat=p_int) tgt
    !ERROR: IOLENGTH variable 'p_int' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    inquire(iolength=p_int) tgt

    associate(assoc_p => p_int)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
      assoc_p = 5
      ! An associate name is not a pointer with the PROTECTED_TARGET attribute,
      ! so C870, C873, and C874 do not apply to it.
      ord_int => assoc_p
      call implicit_sub(assoc_p)
      call sub_no_intent(assoc_p)
    end associate

    associate(assoc_sub => p_sub)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
      assoc_sub%pc => tgt
      !ERROR: 'pc' may not appear in NULLIFY
      !BECAUSE: The target of 'p_sub' with the PROTECTED_TARGET attribute may not be defined
      nullify(assoc_sub%pc)
    end associate

    select type (a => poly_pt)
    type is (integer)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'poly_pt' with the PROTECTED_TARGET attribute may not be defined
      a = 42
    end select

    associate(b => op_int%ptc)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      b = 2
      ord_int => b
    end associate

    associate(a => x_int%ptc)
      !ERROR: 'a' may not be used as a DO variable
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      do a = 1, 2
      end do
      !ERROR: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      read(*, *) (tgt, a = 1, 2)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      a = 3
    end associate

    associate(b_pt => p_int)
      !ERROR: 'b_pt' may not be used as a DO variable
      !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
      do b_pt = 1, 2
      end do
    end associate

    associate(b => op_sub%ptc)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      b%pc => tgt
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      b%a = 1
      !ERROR: 'pc' may not appear in NULLIFY
      !BECAUSE: The target of 'ptc' with the PROTECTED_TARGET attribute may not be defined
      nullify(b%pc)
    end associate

    select type (c => op_poly%cp)
    type is (integer)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'cp' with the PROTECTED_TARGET attribute may not be defined
      c = 5
    end select

    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'p_no_colon' with the PROTECTED_TARGET attribute may not be defined
    p_no_colon = 10

    ! 8.5.16p2: Defining through function result returning PROTECTED_TARGET
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'get_pt' with the PROTECTED_TARGET attribute may not be defined
    get_pt() = 5

    fp => get_pt
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_int => fp()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'fp' with the PROTECTED_TARGET attribute may not be defined
    fp() = 6

    ! C870: Pointer assignment dropping protection
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_int => p_int
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_arr => p_arr(1:2)
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_int => p_sub%a
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_sub => x%ptc
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord_int => get_pt()

    ! C871: Structure constructor dropping protection
    call test_ctor(p_sub, p_arr)

    ! C872: Pointer dummy argument without PROTECTED_TARGET
    !ERROR: dummy argument 'arg=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call sub_ord_ptr(p_int)
    !ERROR: dummy argument 'arg=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call sub_ord_ptr(p_sub%pc)
    !ERROR: dummy argument 'arg=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call sub_ord_ptr(p_sub%ptc2)
    !ERROR: dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call ptr_in(p_arr(1))
    !ERROR: dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call ptr_in(p_sub%a)
    !ERROR: dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call ptr_in(p_sub%pc)
    call sub_pt_ptr(p_sub%pc)

    ! C873: Implicit interface actual argument
    !ERROR: A pointer with the PROTECTED_TARGET attribute (or a subobject thereof) may not be passed to a procedure with an implicit interface
    call implicit_sub(p_int)
    !ERROR: A pointer with the PROTECTED_TARGET attribute (or a subobject thereof) may not be passed to a procedure with an implicit interface
    call implicit_sub(p_sub%a)
    !ERROR: A pointer with the PROTECTED_TARGET attribute (or a subobject thereof) may not be passed to a procedure with an implicit interface
    call implicit_sub(p_arr(1))

    ! C874: Non-pointer dummy argument without INTENT(IN)
    !ERROR: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call sub_no_intent(p_int)
    !ERROR: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call sub_no_intent(p_arr(1))
    !ERROR: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call sub_no_intent(p_sub%a)
    !ERROR: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call sub_no_intent_arr(p_arr(1:2))
    !ERROR: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call sub_value(p_int)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'arg=' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    call sub_intent_out(p_int)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'arg=' is not definable
    !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
    call sub_intent_inout(p_int)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'harvest=' is not definable
    !BECAUSE: The target of 'p_real' with the PROTECTED_TARGET attribute may not be defined
    call random_number(p_real)

    call test_host_assoc()
  contains
    subroutine test_host_assoc()
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'p_int' with the PROTECTED_TARGET attribute may not be defined
      p_int = 100
      !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
      ord_int => p_int
    end subroutine
  end subroutine

  subroutine test_ctor(p, pa)
    type(sub_t), pointer, protected_target :: p
    integer, pointer, protected_target :: pa(:)
    type t_ctor
      type(sub_t), pointer :: c
    end type
    type t_ctor_arr
      integer, pointer :: arr(:)
      integer, pointer :: scalar
    end type
    type(t_ctor) :: res
    type(t_ctor_arr) :: res2
    !ERROR: A structure constructor component must have the PROTECTED_TARGET attribute when the value is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    res = t_ctor(p)
    !ERROR: A structure constructor component must have the PROTECTED_TARGET attribute when the value is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    res = t_ctor(c=p)
    !ERROR: A structure constructor component must have the PROTECTED_TARGET attribute when the value is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    res2 = t_ctor_arr(arr=pa(1:2), scalar=p%pc)
  end subroutine

  subroutine sub_ord_ptr(arg)
    integer, pointer :: arg
  end subroutine

  subroutine ptr_in(a)
    integer, pointer, intent(in) :: a
  end subroutine

  subroutine sub_pt_ptr(arg)
    integer, pointer, protected_target :: arg
  end subroutine

  subroutine sub_no_intent(arg)
    integer :: arg
  end subroutine

  subroutine sub_no_intent_arr(arg)
    integer :: arg(:)
  end subroutine

  subroutine sub_intent_out(arg)
    integer, intent(out) :: arg
  end subroutine

  subroutine sub_intent_inout(arg)
    integer, intent(inout) :: arg
  end subroutine

  subroutine sub_value(arg)
    integer, value :: arg
  end subroutine

  subroutine test_coarray()
    use iso_fortran_env, only: team_type, lock_type
    type(team_type), pointer, protected_target :: tm
    logical, pointer, protected_target :: pl
    type(lock_type), save :: lk[*]
    !ERROR: Team-variable is not definable
    !BECAUSE: The target of 'tm' with the PROTECTED_TARGET attribute may not be defined
    form team(1, tm)
    !ERROR: ACQUIRED_LOCK= variable is not definable
    !BECAUSE: The target of 'pl' with the PROTECTED_TARGET attribute may not be defined
    lock(lk, acquired_lock=pl)
  end subroutine

  subroutine test_c_f_pointer(cp, p)
    use iso_c_binding
    type(c_ptr) :: cp
    integer, pointer, protected_target :: p(:)
    ! C872 is not applied to intrinsic procedures: intrinsic pointer dummy
    ! arguments (e.g. FPTR of C_F_POINTER) only inquire or set association
    ! and never define the target through the pointer.
    call c_f_pointer(cp, p, [1])
  end subroutine

  subroutine test_pt_intent_reassoc(p, tgt)
    integer, pointer, protected_target, intent(in) :: p
    integer, target :: tgt
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    p => tgt
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end subroutine

  subroutine test_proc_ptr_characteristics()
    abstract interface
      subroutine iface_pt(q)
        integer, pointer, protected_target :: q
      end subroutine
      subroutine iface_ord(q)
        integer, pointer :: q
      end subroutine
      function iface_f_pt() result(r)
        integer, pointer, protected_target :: r
      end function
      function iface_f_ord() result(r)
        integer, pointer :: r
      end function
    end interface
    procedure(iface_pt), pointer :: pp_pt
    procedure(iface_ord), pointer :: pp_ord
    procedure(iface_f_pt), pointer :: pp_f_pt
    procedure(iface_f_ord), pointer :: pp_f_ord

    ! Procedure pointer assignment both directions accepted.
    ! The unsafe direction (interface expecting a PROTECTED_TARGET dummy
    ! associated with an ordinary pointer target) is permitted because
    ! PROTECTED_TARGET is not a procedure characteristic in Fortran 2028
    ! (J3/26-007r1 15.3.2.2 and 15.3.3).
    pp_pt => pp_ord
    pp_ord => pp_pt

    ! Function result compatibility both directions accepted
    pp_f_pt => pp_f_ord
    pp_f_ord => pp_f_pt
    pp_f_pt => func_ord
    pp_f_ord => func_pt

    call call_dummy(pp_ord)
    call call_dummy(pp_pt)
  contains
    subroutine call_dummy(dproc)
      procedure(iface_pt) :: dproc
    end subroutine
  end subroutine

  subroutine test_do_concurrent()
    integer, pointer, protected_target :: p
    integer :: i
    !ERROR: 'p' may not appear in a locality-spec because it is not definable
    !BECAUSE: The target of 'p' with the PROTECTED_TARGET attribute may not be defined
    do concurrent (i=1:2) local(p)
    end do
  end subroutine
end module

module m_pt_rename_src
  integer, target :: tgt = 1
  integer, pointer, protected_target :: src_pt
end module

module m_pt_rename_test
  use m_pt_rename_src, only: ren_pt => src_pt
contains
  subroutine test_rename()
    integer, pointer :: ord
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of 'ren_pt' with the PROTECTED_TARGET attribute may not be defined
    ren_pt = 1
    !ERROR: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    ord => ren_pt
  end subroutine
end module

module m_generic_pt
  type s_gen
    integer, pointer :: pc
  end type
  interface gen_elem
    module procedure s_arr
    module procedure s_elem
  end interface
  interface gen_plain
    module procedure s_int
    module procedure s_real
  end interface
  interface gen_ptr
    module procedure s_ord_ptr
    module procedure s_real_ptr
  end interface
contains
  subroutine s_arr(x)
    integer :: x(:)
    x = 0
  end subroutine
  elemental subroutine s_elem(x)
    integer, intent(in) :: x
  end subroutine
  subroutine s_int(x)
    integer :: x
  end subroutine
  subroutine s_real(x)
    real :: x
  end subroutine
  subroutine s_ord_ptr(x)
    integer, pointer :: x
  end subroutine
  subroutine s_real_ptr(x)
    real, pointer :: x
  end subroutine
  subroutine test_generic()
    integer, pointer, protected_target :: p(:)
    integer, pointer, protected_target :: p_scalar
    type(s_gen), pointer, protected_target :: p_sub
    ! C874: elemental vs nonelemental resolves to nonelemental specific, which violates C874
    !ERROR: Actual argument associated with dummy argument 'x=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call gen_elem(p)
    ! C874: plain generic reports C874 on matching specific, not "no specific matches"
    !ERROR: Actual argument associated with dummy argument 'x=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
    call gen_plain(p_scalar)
    ! C872: generic interface reports C872 on matching specific, not "no specific matches"
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
    call gen_ptr(p_sub%pc)
  end subroutine
end module

module m_tbp_override
  type base
  contains
    procedure, nopass :: tb => tb_base
  end type
  type, extends(base) :: ext
  contains
    procedure, nopass :: tb => tb_ext
  end type
contains
  subroutine tb_base(a)
    integer, pointer :: a
  end subroutine
  subroutine tb_ext(a)
    integer, pointer, protected_target :: a
  end subroutine
end module

module m_sep_interface
  interface
    module subroutine sep(a)
      integer, pointer :: a
    end subroutine
    ! 26-007r1 15.3.2.2: PROTECTED_TARGET is not a characteristic, so
    ! the interface may declare it even if the module procedure body does not.
    module subroutine sep_unsafe(a)
      integer, pointer, protected_target :: a
    end subroutine
  end interface
end module

submodule (m_sep_interface) s_sep_interface
contains
  module subroutine sep(a)
    integer, pointer, protected_target :: a
  end subroutine
  module subroutine sep_unsafe(a)
    integer, pointer :: a
    a = 999
  end subroutine
end submodule

module m_select_rank_default
contains
  subroutine ord_ptr_ar(x)
    integer, pointer :: x(..)
  end subroutine
  subroutine ord_ptr_1d(x)
    integer, pointer :: x(:)
  end subroutine
  subroutine no_intent_ar(x)
    integer :: x(..)
  end subroutine
  subroutine test_select_rank_default(pr)
    integer, pointer, protected_target :: pr(..)
    integer, pointer :: ord1(:)
    select rank (r => pr)
    rank (1)
      ! J3/26-007r1 11.1.12.3p3: r is a pointer without PROTECTED_TARGET,
      ! so C870 and C872 do not apply to it; it is still not definable.
      ord1 => r
      call ord_ptr_1d(r)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of 'pr' with the PROTECTED_TARGET attribute may not be defined
      r(1) = 0
    rank default
      !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
      call ord_ptr_ar(r)
      !ERROR: Actual argument associated with dummy argument 'x=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
      call no_intent_ar(r)
      !ERROR: A pointer with the PROTECTED_TARGET attribute may not be an allocate-object in a DEALLOCATE statement
      deallocate(r)
    end select
  end subroutine
end module
