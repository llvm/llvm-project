! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test for component-wise initialization of derived types to avoid
! generating large, fully-initialized global templates.

module my_types
  ! Clean internal type: contains only allocatable and pointer components, 
  ! without any explicit initialization.
  type :: InnerClean
    real, pointer :: p
    real, allocatable :: arr(:)
  end type InnerClean

  ! ============================================================================
  ! Test Case 1: Type meeting all criteria for component-wise init (Target)
  ! No arrays of derived types, no explicit initialization.
  ! ============================================================================
  type :: TargetType
    real, pointer :: q
    real(8) :: uninit_buffer(80, 100)
    type(InnerClean) :: nested_scalar
  end type TargetType

  ! ============================================================================
  ! Test Case 2: Type triggering fallback A (contains an array of derived type)
  ! ============================================================================
  type :: FallbackArrayType
    type(InnerClean) :: nested_arr(10)
  end type FallbackArrayType

  ! ============================================================================
  ! Test Case 3: Type triggering fallback B (contains explicit initialization)
  ! ============================================================================
  type :: FallbackExplicitType
    real, pointer :: p
    integer :: flag = 999
  end type FallbackExplicitType

  ! ============================================================================
  ! Test Case 4: Type triggering fallback C (contains explicit pointer initialization)
  ! ============================================================================
  type :: ProcPointerType
    real, pointer :: p
    procedure(), pointer, nopass :: pp => null()
  end type ProcPointerType
end module my_types


! ------------------------------------------------------------------------------
! Test 1: FIR generation for component-wise initialization
! ------------------------------------------------------------------------------
subroutine test_target()
  use my_types
  type(TargetType) :: my_var
  call do_something_target(my_var)
end subroutine test_target

! CHECK-LABEL: func.func @_QPtest_target()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QMmy_typesTtargettype{{.*}}>
! CHECK: %[[MY_VAR:.*]] = fir.declare %[[ALLOCA]]
! CHECK: %[[Q_ADDR:.*]] = fir.coordinate_of %[[MY_VAR]], q
! CHECK: fir.store %{{.*}} to %[[Q_ADDR]]
! CHECK: %[[NESTED_ADDR:.*]] = fir.coordinate_of %[[MY_VAR]], nested_scalar
! CHECK: %[[P_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], p
! CHECK: fir.store %{{.*}} to %[[P_ADDR]]
! CHECK: %[[ARR_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], arr
! CHECK: fir.store %{{.*}} to %[[ARR_ADDR]]
! CHECK-NOT: fir.call @_FortranAInitialize
! CHECK-NOT: fir.global internal @_QQ_QMmy_typesTtargettype.DerivedInit


! ------------------------------------------------------------------------------
! Test 2: Fallback mechanism for arrays of derived types
! ------------------------------------------------------------------------------
subroutine test_fallback_array()
  use my_types
  type(FallbackArrayType) :: var_array
  call do_something_array(var_array)
end subroutine test_fallback_array

! CHECK-LABEL: func.func @_QPtest_fallback_array()
! CHECK: %[[ALLOCA_ARR:.*]] = fir.alloca !fir.type<_QMmy_typesTfallbackarraytype{{.*}}>
! CHECK: %[[VAR_ARRAY:.*]] = fir.declare %[[ALLOCA_ARR]]
! CHECK: %[[GLOBAL_INIT_ARR:.*]] = fir.address_of(@_QQ_QMmy_typesTfallbackarraytype.DerivedInit)
! CHECK: fir.copy %[[GLOBAL_INIT_ARR]] to %[[VAR_ARRAY]]


! ------------------------------------------------------------------------------
! Test 3: Fallback mechanism for explicit initialization
! ------------------------------------------------------------------------------
subroutine test_fallback_explicit()
  use my_types
  type(FallbackExplicitType) :: var_explicit
  call do_something_explicit(var_explicit)
end subroutine test_fallback_explicit

! CHECK-LABEL: func.func @_QPtest_fallback_explicit()
! CHECK: %[[ALLOCA_EXP:.*]] = fir.alloca !fir.type<_QMmy_typesTfallbackexplicittype{{.*}}>
! CHECK: %[[VAR_EXPLICIT:.*]] = fir.declare %[[ALLOCA_EXP]]
! CHECK: %[[GLOBAL_INIT_EXP:.*]] = fir.address_of(@_QQ_QMmy_typesTfallbackexplicittype.DerivedInit)
! CHECK: fir.copy %[[GLOBAL_INIT_EXP]] to %[[VAR_EXPLICIT]]


! ------------------------------------------------------------------------------
! Test 4: Procedure pointers
! Procedure pointers default initialized with '=> null()' trigger the explicit 
! initialization fallback safely.
! ------------------------------------------------------------------------------
subroutine test_proc_pointer()
  use my_types
  type(ProcPointerType) :: var_proc
  call do_something_proc(var_proc)
end subroutine test_proc_pointer

! CHECK-LABEL: func.func @_QPtest_proc_pointer()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QMmy_typesTprocpointertype{{.*}}>
! CHECK: %[[VAR_DECL:.*]] = fir.declare %[[ALLOCA]] {{.*}}
! CHECK: %[[GLOBAL_INIT_PROC:.*]] = fir.address_of(@_QQ_QMmy_typesTprocpointertype.DerivedInit)
! CHECK: fir.copy %[[GLOBAL_INIT_PROC]] to %[[VAR_DECL]]
