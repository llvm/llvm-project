! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=52
! OpenMP Version 5.2
! 14.1 Interop construct
! To check various semantic errors for inteorp construct.

SUBROUTINE test_interop_01()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: Each interop-var may be specified for at most one action-clause of each INTEROP construct.
  !$OMP INTEROP INIT(TARGETSYNC,TARGET: obj) USE(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_01

SUBROUTINE test_interop_02()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: Each interop-type may be specified at most once.
  !$OMP INTEROP INIT(TARGETSYNC,TARGET,TARGETSYNC: obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_02

SUBROUTINE test_interop_03()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: A DEPEND clause can only appear on the directive if the interop-type includes TARGETSYNC
  !$OMP INTEROP INIT(TARGET: obj) DEPEND(INOUT: obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_03

SUBROUTINE test_interop_04()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(TARGETSYNC: obj)
  !ERROR: The DESTROY clause on an INTEROP construct must specify an interop variable
  !$OMP INTEROP DESTROY
  PRINT *, 'pass'
END SUBROUTINE test_interop_04

SUBROUTINE test_interop_05()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !ERROR: The interop variable in an INTEROP construct must be a scalar variable
  !$OMP INTEROP INIT(TARGET: arr)
  PRINT *, 'pass'
END SUBROUTINE test_interop_05

SUBROUTINE test_interop_06()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !ERROR: The interop variable in an INTEROP construct must be a scalar variable
  !$OMP INTEROP USE(arr(1:5))
  PRINT *, 'pass'
END SUBROUTINE test_interop_06

SUBROUTINE test_interop_07()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !ERROR: The interop variable in an INTEROP construct must be a scalar variable
  !$OMP INTEROP DESTROY(arr)
  PRINT *, 'pass'
END SUBROUTINE test_interop_07

SUBROUTINE test_interop_08()
  REAL(8) :: x
  !ERROR: The interop variable in an INTEROP construct must be a scalar integer variable of kind omp_interop_kind
  !$OMP INTEROP INIT(TARGET: x)
  PRINT *, 'pass'
END SUBROUTINE test_interop_08

SUBROUTINE test_interop_09()
  INTEGER(4) :: obj
  !ERROR: The interop variable in an INTEROP construct must be a scalar integer variable of kind omp_interop_kind
  !$OMP INTEROP USE(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_09

SUBROUTINE test_interop_10()
  INTEGER(4) :: obj
  !ERROR: The interop variable in an INTEROP construct must be a scalar integer variable of kind omp_interop_kind
  !$OMP INTEROP DESTROY(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_10

! An array element is a valid interop-var (scalar, of omp_interop_kind) and
! must be accepted on all action clauses.
SUBROUTINE test_interop_11()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !$OMP INTEROP INIT(TARGET: arr(1))
  !$OMP INTEROP USE(arr(1))
  !$OMP INTEROP DESTROY(arr(1))
  PRINT *, 'pass'
END SUBROUTINE test_interop_11

! Uniqueness is compared per complete designator: distinct array elements or
! structure components are different interop-vars and must be accepted.
SUBROUTINE test_interop_12()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !$OMP INTEROP INIT(TARGETSYNC: arr(1)) USE(arr(2))
  PRINT *, 'pass'
END SUBROUTINE test_interop_12

! Repeating the same designator in two action-clauses is a duplicate.
SUBROUTINE test_interop_13()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: arr(10)
  !ERROR: Each interop-var may be specified for at most one action-clause of each INTEROP construct.
  !$OMP INTEROP INIT(TARGETSYNC: arr(1)) USE(arr(1))
  PRINT *, 'pass'
END SUBROUTINE test_interop_13

! init and destroy store the new handle through the interop-var, so it must be
! a definable variable, not a constant or other non-definable entity.
SUBROUTINE test_interop_14()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND), PARAMETER :: handle = 0
  !ERROR: The interop variable in an INTEROP construct must be a definable variable
  !BECAUSE: 'handle' is not a variable
  !$OMP INTEROP INIT(TARGETSYNC: handle)
  PRINT *, 'pass'
END SUBROUTINE test_interop_14

SUBROUTINE test_interop_15(obj)
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND), INTENT(IN) :: obj
  !ERROR: The interop variable in an INTEROP construct must be a definable variable
  !BECAUSE: 'obj' is an INTENT(IN) dummy argument
  !$OMP INTEROP DESTROY(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_15

! A use clause only reads the handle, so it does not require a definable
! variable, but the interop-var must still be a variable (not a constant).
SUBROUTINE test_interop_16()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND), PARAMETER :: handle = 0
  !ERROR: The interop variable in an INTEROP construct must be a variable
  !$OMP INTEROP USE(handle)
  PRINT *, 'pass'
END SUBROUTINE test_interop_16

! An INTENT(IN) dummy holding an initialized handle is a valid use interop-var.
SUBROUTINE test_interop_17(obj)
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND), INTENT(IN) :: obj
  !$OMP INTEROP USE(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_17

! An interop construct must have at least one action-clause; a construct with
! only a device (or other non-action) clause is invalid.
SUBROUTINE test_interop_18(dev)
  INTEGER :: dev
  !ERROR: At least one action-clause (INIT, USE, or DESTROY) must appear on the INTEROP construct
  !$OMP INTEROP DEVICE(dev)
  PRINT *, 'pass'
END SUBROUTINE test_interop_18

! A non-designator interop-var -- a function reference or a reserved locator
! such as omp_all_memory -- is not a variable and must be rejected (otherwise
! it bypasses validation and asserts in lowering).
SUBROUTINE test_interop_19()
  INTEGER(8), EXTERNAL :: get_handle
  !ERROR: The interop variable in an INTEROP construct must be a variable
  !$OMP INTEROP USE(get_handle())
  PRINT *, 'pass'
END SUBROUTINE test_interop_19

SUBROUTINE test_interop_20()
  !ERROR: The interop variable in an INTEROP construct must be a variable
  !$OMP INTEROP USE(omp_all_memory)
  PRINT *, 'pass'
END SUBROUTINE test_interop_20

! The same rejection applies to init and destroy (shared code path).
SUBROUTINE test_interop_21()
  INTEGER(8), EXTERNAL :: get_handle
  !ERROR: The interop variable in an INTEROP construct must be a variable
  !$OMP INTEROP INIT(TARGET: get_handle())
  PRINT *, 'pass'
END SUBROUTINE test_interop_21

SUBROUTINE test_interop_22()
  !ERROR: The interop variable in an INTEROP construct must be a variable
  !$OMP INTEROP DESTROY(omp_all_memory)
  PRINT *, 'pass'
END SUBROUTINE test_interop_22
