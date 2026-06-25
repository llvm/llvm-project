! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! A list item that appears in a lastprivate clause with a conditional modifier
! must be a scalar variable.

subroutine foo()
  integer :: s, i
  character(len=8) :: c
  character(len=:), allocatable :: dc
  integer :: arr(10)
  integer :: mat(3, 3)
  type t
    integer :: a
  end type
  type(t) :: dt
  class(t), allocatable :: poly
  integer, allocatable :: alloc
  integer, pointer :: ptr
  integer, target :: tgt
  real :: r
  complex :: z
  logical :: lg
  class(*), allocatable :: up
  character(len=8), allocatable :: ca
  character(len=8), pointer :: cp
  type(t), allocatable :: da
  type(t), pointer :: dp

  ! Scalar intrinsic list items of every accepted category are allowed.
  !$omp do lastprivate(conditional: s)
  do i = 1, 100
    if (mod(i, 2) == 0) s = i
  enddo
  !$omp end do

  !$omp do lastprivate(conditional: r)
  do i = 1, 100
    if (mod(i, 2) == 0) r = i
  enddo
  !$omp end do

  !$omp do lastprivate(conditional: z)
  do i = 1, 100
    if (mod(i, 2) == 0) z = cmplx(i, i)
  enddo
  !$omp end do

  !$omp do lastprivate(conditional: lg)
  do i = 1, 100
    if (mod(i, 2) == 0) lg = mod(i, 3) == 0
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'c' is not
  !$omp do lastprivate(conditional: c)
  do i = 1, 100
    if (mod(i, 2) == 0) c = 'even'
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'arr' is not
  !$omp do lastprivate(conditional: arr)
  do i = 1, 100
    if (mod(i, 2) == 0) arr(1) = i
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'mat' is not
  !$omp do lastprivate(conditional: mat)
  do i = 1, 100
    if (mod(i, 2) == 0) mat(1, 1) = i
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'dt' is not
  !$omp do lastprivate(conditional: dt)
  do i = 1, 100
    if (mod(i, 2) == 0) dt%a = i
  enddo
  !$omp end do

!ERROR: A POINTER or ALLOCATABLE list item is not yet supported by Flang in a LASTPRIVATE clause with the CONDITIONAL modifier, 'alloc'
  !$omp do lastprivate(conditional: alloc)
  do i = 1, 100
    if (mod(i, 2) == 0) alloc = i
  enddo
  !$omp end do

  ptr => tgt
!ERROR: A POINTER or ALLOCATABLE list item is not yet supported by Flang in a LASTPRIVATE clause with the CONDITIONAL modifier, 'ptr'
  !$omp do lastprivate(conditional: ptr)
  do i = 1, 100
    if (mod(i, 2) == 0) ptr = i
  enddo
  !$omp end do

  ! A polymorphic entity is not a scalar variable of intrinsic type.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'poly' is not
  !$omp do lastprivate(conditional: poly)
  do i = 1, 100
    if (mod(i, 2) == 0) poly = t(i)
  enddo
  !$omp end do

  ! A deferred-length character is still character type and is excluded.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'dc' is not
  !$omp do lastprivate(conditional: dc)
  do i = 1, 100
    if (mod(i, 2) == 0) dc = 'x'
  enddo
  !$omp end do

  ! Unlimited polymorphic is not a scalar variable of intrinsic type.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'up' is not
  !$omp do lastprivate(conditional: up)
  do i = 1, 100
  enddo
  !$omp end do

  ! An ALLOCATABLE character is character type; the type check fires before
  ! the POINTER/ALLOCATABLE check.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'ca' is not
  !$omp do lastprivate(conditional: ca)
  do i = 1, 100
  enddo
  !$omp end do

  ! A POINTER character is likewise excluded by the type check.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'cp' is not
  !$omp do lastprivate(conditional: cp)
  do i = 1, 100
  enddo
  !$omp end do

  ! An ALLOCATABLE derived type is excluded by the type check.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'da' is not
  !$omp do lastprivate(conditional: da)
  do i = 1, 100
  enddo
  !$omp end do

  ! A POINTER derived type is excluded by the type check.
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'dp' is not
  !$omp do lastprivate(conditional: dp)
  do i = 1, 100
  enddo
  !$omp end do
end


! A common block is not a scalar variable (it has no type), so it is rejected
! by the conditional scalar-variable restriction.
subroutine bar()
  integer :: i
  integer :: gi
  real :: gr
  common /cb/ gi, gr
!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable with intrinsic type, as defined by the Fortran language, excluding character type, but 'cb' is not
  !$omp do lastprivate(conditional: /cb/)
  do i = 1, 100
  enddo
  !$omp end do
end
