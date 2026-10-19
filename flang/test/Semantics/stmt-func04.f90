! RUN: %python %S/test_errors.py %s %flang_fc1
! F2023 19.4 p2: a statement function dummy argument name may be the same as an
! accessible global identifier or local identifier of class (1) only if that
! name is a scalar variable.  Only names made visible by the scoping unit
! itself can conflict: per 19.5.1.4 p2 item (11), the name's appearance as a
! statement function dummy argument renders any host entity of that name
! inaccessible by host association, and a global entity to which the scoping
! unit makes no other reference is not accessible in it either.

! Clashes within the statement function's own scoping unit.
subroutine local_clashes
  real, external :: extf       ! external procedure
  real, parameter :: namedc = 1.0
  real :: arr(10)              ! array
  type t; end type             ! derived type
  real :: scalarvar            ! scalar variable (legal to shadow)

  !ERROR: The name 'extf' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  f1(extf) = extf + 1
  !ERROR: The name 'namedc' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  f2(namedc) = namedc + 1
  !ERROR: The name 'arr' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  f3(arr) = arr + 1
  !ERROR: The name 't' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  f4(t) = 1
  f5(scalarvar) = scalarvar + 1 ! ok: scalar variable shadowing is permitted
end subroutine

! No clashes with host entities (module scope): the dummy argument's
! appearance blocks host association (19.5.1.4 p2 item (11)), so the host
! entities are inaccessible here and the dummies are implicitly typed.
module m
  integer :: hostarr(10)
  integer :: hostscalar
  integer, parameter :: hostconst = 3
contains
  subroutine host_no_clashes
    g1(hostarr) = hostarr + 1 ! ok: host's 'hostarr' is inaccessible here
    g2(hostconst) = hostconst + 1 ! ok: host's 'hostconst' is inaccessible here
    g3(hostscalar) = hostscalar + 1 ! ok: host scalar variable
  end subroutine
end module

! Same for grandparent host entities (internal procedure).
program p
  integer :: grandarr(5)
  integer :: grandscalar
contains
  subroutine grand_no_clashes
    h1(grandarr) = grandarr + 1 ! ok: host's 'grandarr' is inaccessible here
    h2(grandscalar) = grandscalar + 1 ! ok: grandparent scalar variable
  end subroutine
end program

! Clashes with USE-associated identifiers.
module m_used
  integer :: usearr(10)
  integer :: usescalar
  integer, parameter :: useconst = 5
end module
subroutine use_clashes
  use m_used
  !ERROR: The name 'usearr' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  k1(usearr) = usearr + 1
  !ERROR: The name 'useconst' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  k2(useconst) = useconst + 1
  k3(usescalar) = usescalar + 1 ! ok: USE-associated scalar variable
end subroutine

! No clash with a global-scope program unit to which the scoping unit makes
! no other reference: that global identifier is not accessible in it.
real function global_func(x)
  real :: x
  global_func = x
end function
subroutine global_no_clashes
  p1(global_func) = global_func + 1 ! ok: 'global_func' is not accessible here
end subroutine

! Likewise for a bind(c) global subprogram.
real function bindc_global_func() bind(c)
  bindc_global_func = 1.0
end function
subroutine bindc_global_no_clashes
  r1(bindc_global_func) = bindc_global_func + 1 ! ok: not accessible here
end subroutine

! Clashes with bind(c) external declared via local explicit interface block.
subroutine bindc_interface_clashes
  interface
    real function c_iface_func() bind(c)
    end function
  end interface
  !ERROR: The name 'c_iface_func' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  r2(c_iface_func) = c_iface_func + 1
end subroutine

! bind(c) USE-associated variables: scalar is ok; array is an error.
module m_bindc
  real, bind(c) :: c_scalar
  real, bind(c) :: c_array(4)
end module
subroutine bindc_use_var_clashes
  use m_bindc
  r3(c_scalar) = c_scalar + 1  ! ok: bind(c) scalar variable
  !ERROR: The name 'c_array' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  r4(c_array) = c_array + 1
end subroutine

! bind(c) USE-associated module procedure (error).
module m_bindc_proc
contains
  real function c_mod_func() bind(c)
    c_mod_func = 1.0
  end function
end module
subroutine bindc_use_proc_clashes
  use m_bindc_proc
  !ERROR: The name 'c_mod_func' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  r5(c_mod_func) = c_mod_func + 1
end subroutine

! A host EXTERNAL declaration does not make the name accessible in an inner
! subprogram that uses it as a statement function dummy argument; host
! association of the name is blocked throughout that subprogram (19.5.1.4 p2
! item (11)), so the dummy is implicitly typed there (default real).  If host
! association leaked through, 'pred * pred' would be a type error.
module m_host_external
  logical, external :: pred
contains
  subroutine eval(arg)
    real :: sq
    sq(pred) = pred * pred ! ok: 'pred' is an implicitly typed real dummy here
    print *, sq(arg)
  end subroutine
  subroutine other
    logical :: q
    q = pred() ! ok: host association of 'pred' is severed only in 'eval'
    print *, q
  end subroutine
end module

! F2023 C8106: an explicitly imported host name, or any host name made
! accessible by IMPORT, ALL, may not be hidden, and a statement function
! dummy argument of that name would hide it.  There is no scalar-variable
! exception here.  A plain IMPORT does not protect names from being hidden
! (8.8 p4).
subroutine import_host
  real :: val
contains
  subroutine only_import
    import, only: val
    !ERROR: 'val' from host may not be hidden by a statement function dummy argument
    w1(val) = val + 1
  end subroutine
  subroutine named_import
    import :: val
    !ERROR: 'val' from host may not be hidden by a statement function dummy argument
    w2(val) = val + 1
  end subroutine
  subroutine all_import
    import, all
    !ERROR: 'val' from host may not be hidden by a statement function dummy argument
    w3(val) = val + 1
  end subroutine
  subroutine plain_import
    import
    w4(val) = val + 1 ! ok: plain IMPORT tolerates hiding (8.8 p4)
  end subroutine
end subroutine

! Boundary cases that remain errors: the enclosing subprogram's own name and
! an ENTRY name of the same subprogram are visible in the subprogram itself.
subroutine self_clash
  real :: f
  !ERROR: The name 'self_clash' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  f(self_clash) = self_clash + 1
end subroutine
subroutine entry_clash
  real :: g
  !ERROR: The name 'ent' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  g(ent) = ent + 1
  return
entry ent
end subroutine

! A common block name is explicitly excepted by 19.4 p2.
subroutine common_carveout
  real :: y, f
  common /cb/ y
  f(cb) = cb + 1 ! ok: common block name carve-out
end subroutine
