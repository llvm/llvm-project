!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

! A BIND(C) interface with an assumed-shape or assumed-rank dummy argument
! is only interoperable because the actual argument is passed using a
! Fortran 2018 CFI descriptor (CFI_cdesc_t); a C/C++ side written against
! an older calling convention that expects a bare address will not agree
! with this ABI. Warn about this case under -pedantic (UsageWarning
! BindCArrayDescriptor is off by default; see bind-c21.f90 for the silent
! default), unless the BIND(C) binding name (explicit or default) starts
! with '_', which marks an implementation-internal interface rather than
! genuine external interoperability.

subroutine assumedShape(a)
  interface
    subroutine cFunc(a) bind(c)
      !PORTABILITY: Dummy argument 'a' of BIND(C) interface 'cfunc' is assumed-shape; the C/C++ side must accept a Fortran 2018 CFI descriptor (CFI_cdesc_t), not a bare address, which may not match an external interface written for an older calling convention [-Wbind-c-array-descriptor]
      real, intent(in) :: a(:)
    end subroutine
  end interface
  real :: a(10)
  call cFunc(a)
end subroutine

subroutine assumedRank(a)
  interface
    subroutine cFunc2(a) bind(c)
      !PORTABILITY: Dummy argument 'a' of BIND(C) interface 'cfunc2' is assumed-rank; the C/C++ side must accept a Fortran 2018 CFI descriptor (CFI_cdesc_t), not a bare address, which may not match an external interface written for an older calling convention [-Wbind-c-array-descriptor]
      real, intent(in) :: a(..)
    end subroutine
  end interface
  real :: a(10)
  call cFunc2(a)
end subroutine

subroutine explicitShapeUnaffected(a, n)
  interface
    subroutine cFunc3(a, n) bind(c)
      integer, value :: n
      real, intent(in) :: a(n)
    end subroutine
  end interface
  integer :: n
  real :: a(10)
  n = 10
  call cFunc3(a, n)
end subroutine

subroutine explicitBindingName(a)
  interface
    subroutine cFunc4(a) bind(c, name="my_c_func")
      !PORTABILITY: Dummy argument 'a' of BIND(C) interface 'cfunc4' is assumed-shape; the C/C++ side must accept a Fortran 2018 CFI descriptor (CFI_cdesc_t), not a bare address, which may not match an external interface written for an older calling convention [-Wbind-c-array-descriptor]
      real, intent(in) :: a(:)
    end subroutine
  end interface
  real :: a(10)
  call cFunc4(a)
end subroutine

subroutine implementationInternalUnaffected(a)
  ! A binding name starting with '_' (here, the implicit default, since no
  ! NAME= is given) identifies a compiler-internal interface rather than a
  ! genuine external C/C++ interoperability interface, so this is not
  ! warned about: such interfaces are written to receive the CFI
  ! descriptor this warning is about.
  interface
    subroutine _internalFunc(a) bind(c)
      real, intent(in) :: a(:)
    end subroutine
  end interface
  real :: a(10)
  call _internalFunc(a)
end subroutine
