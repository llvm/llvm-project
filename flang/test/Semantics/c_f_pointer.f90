! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Enforce 18.2.3.3

program test
  use iso_c_binding, only: c_ptr, c_f_pointer
  type(c_ptr) :: scalarC, arrayC(1)
  type :: with_pointer
    integer, pointer :: p
  end type
  type(with_pointer) :: coindexed[*]
  integer, pointer :: scalarIntF, arrayIntF(:), multiDimIntF(:,:)
  character(len=:), pointer :: charDeferredF
  integer :: j
  integer, dimension(2, 2) :: rankTwoArray
  class(*), pointer :: unlimited
  type :: notBindCType
    integer :: n
  end type
  type(notBindCType), pointer :: notBindC
  character(2), pointer :: c2ptr
  rankTwoArray = reshape([1, 2, 3, 4], shape(rankTwoArray))
  call c_f_pointer(scalarC, scalarIntF) ! ok
  call c_f_pointer(scalarC, arrayIntF, [1_8]) ! ok
  call c_f_pointer(shape=[1_8], cptr=scalarC, fptr=arrayIntF) ! ok
  call c_f_pointer(scalarC, shape=[1_8], fptr=arrayIntF) ! ok
  !ERROR: A positional actual argument may not appear after any keyword arguments
  call c_f_pointer(scalarC, fptr=arrayIntF, [1_8])
  !ERROR: CPTR= argument to C_F_POINTER() must be a C_PTR
  call c_f_pointer(j, scalarIntF)
  !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
  call c_f_pointer(arrayC, scalarIntF)
  !ERROR: SHAPE= argument to C_F_POINTER() must appear when FPTR= is an array
  call c_f_pointer(scalarC, arrayIntF)
  !ERROR: SHAPE= argument to C_F_POINTER() may not appear when FPTR= is scalar
  call c_f_pointer(scalarC, scalarIntF, [1_8])
  !ERROR: FPTR= argument to C_F_POINTER() may not have a deferred type parameter
  call c_f_pointer(scalarC, charDeferredF)
  !ERROR: FPTR= argument to C_F_POINTER() may not be a coindexed object
  !ERROR: A coindexed object may not be a pointer target
  call c_f_pointer(scalarC, coindexed[0]%p)
  !ERROR: FPTR= argument to C_F_POINTER() must have a type
  call c_f_pointer(scalarC, null())
  !ERROR: SHAPE= argument to C_F_POINTER() must have size equal to the rank of FPTR=
  call c_f_pointer(scalarC, multiDimIntF, shape=[1_8])
  !ERROR: SHAPE= argument to C_F_POINTER() must be a rank-one array.
  call c_f_pointer(scalarC, multiDimIntF, shape=rankTwoArray)
  !WARNING: FPTR= argument to C_F_POINTER() should not be unlimited polymorphic
  call c_f_pointer(scalarC, unlimited)
  !WARNING: FPTR= argument to C_F_POINTER() should not have a derived type that is not BIND(C)
  call c_f_pointer(scalarC, notBindC)
  !WARNING: FPTR= argument to C_F_POINTER() should not have the non-interoperable intrinsic type CHARACTER(KIND=1,LEN=2_8)
  call c_f_pointer(scalarC, c2ptr)
end program
