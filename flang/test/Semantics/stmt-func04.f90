! RUN: %python %S/test_errors.py %s %flang_fc1
! F2023 19.4 p2: a statement function dummy argument name may be the same as an
! accessible global identifier or local identifier of class (1) only if that
! name is a scalar variable.

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

! Clashes with host-associated identifiers (module scope).
module m
  integer :: hostarr(10)
  integer :: hostscalar
  integer, parameter :: hostconst = 3
contains
  subroutine host_clashes
    !ERROR: The name 'hostarr' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
    g1(hostarr) = hostarr + 1
    !ERROR: The name 'hostconst' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
    g2(hostconst) = hostconst + 1
    g3(hostscalar) = hostscalar + 1 ! ok: host scalar variable
  end subroutine
end module

! Clashes with grandparent-associated identifiers (internal procedure).
program p
  integer :: grandarr(5)
  integer :: grandscalar
contains
  subroutine grand_clashes
    !ERROR: The name 'grandarr' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
    h1(grandarr) = grandarr + 1
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

! Clashes with global-scope program units.
real function global_func(x)
  real :: x
  global_func = x
end function
subroutine global_clashes
  !ERROR: The name 'global_func' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  p1(global_func) = global_func + 1
end subroutine

! Clashes with bind(c) global subprogram (exercises global scope path).
real function bindc_global_func() bind(c)
  bindc_global_func = 1.0
end function
subroutine bindc_global_clashes
  !ERROR: The name 'bindc_global_func' of a statement function dummy argument may not be the same as an accessible name unless that name is a scalar variable
  r1(bindc_global_func) = bindc_global_func + 1
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
