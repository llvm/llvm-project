! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test for precise component-wise initialization of derived types
! to avoid generating large, fully-initialized global templates (sparse initialization).

module my_types
  ! Clean internal type: contains only allocatable and pointer components, 
  ! without any explicit initialization.
  type :: InnerClean
    real, pointer :: p
    real, allocatable :: arr(:)
  end type InnerClean

  ! ============================================================================
  ! Test Case 1: Type meeting all criteria for precise component-wise init (Target)
  ! No arrays of derived types, no explicit initialization.
  ! ============================================================================
  type :: TargetType
    real, pointer :: q
    real(8) :: uninit_buffer(80, 100)      ! Large uninitialized array; should be bypassed.
    type(InnerClean) :: nested_scalar      ! Scalar derived type; should be expanded recursively.
  end type TargetType

  ! ============================================================================
  ! Test Case 2: Type triggering fallback A (contains an array of derived type)
  ! ============================================================================
  type :: FallbackArrayType
    type(InnerClean) :: nested_arr(10)     ! Array of derived type; must trigger fallback.
  end type FallbackArrayType

  ! ============================================================================
  ! Test Case 3: Type triggering fallback B (contains explicit initialization)
  ! ============================================================================
  type :: FallbackExplicitType
    real, pointer :: p
    integer :: flag = 999                  ! Explicit initialization; must trigger gatekeeper!
  end type FallbackExplicitType
end module my_types


! ------------------------------------------------------------------------------
! Test 1: FIR generation for precise component-wise initialization
! ------------------------------------------------------------------------------
subroutine test_target()
  use my_types
  type(TargetType) :: my_var
  call do_something_target(my_var)
end subroutine test_target

! CHECK-LABEL: func.func @_QPtest_target()
! Match the alloca and the subsequent declare
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QMmy_typesTtargettype{{.*}}>
! CHECK: %[[MY_VAR:.*]] = fir.declare %[[ALLOCA]]

! 1. Check initialization of the outermost pointer 'q'
! CHECK: %[[Q_ADDR:.*]] = fir.coordinate_of %[[MY_VAR]], q
! CHECK: fir.store %{{.*}} to %[[Q_ADDR]]

! 2. Check recursive precise initialization of the scalar derived type 'nested_scalar'
! CHECK: %[[NESTED_ADDR:.*]] = fir.coordinate_of %[[MY_VAR]], nested_scalar

! -> Check internal pointer 'p'
! CHECK: %[[P_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], p
! CHECK: fir.store %{{.*}} to %[[P_ADDR]]

! -> Check internal allocatable array 'arr'
! CHECK: %[[ARR_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], arr
! CHECK: fir.store %{{.*}} to %[[ARR_ADDR]]

! CRITICAL 1: Ensure NO expensive runtime calls are generated.
! CHECK-NOT: fir.call @_FortranAInitialize

! CRITICAL 2: Ensure NO large global initialization templates are generated for TargetType.
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
! 
! Ensure FallbackArrayType falls back to the highly optimized memcpy from a global template,
! entirely avoiding the runtime loop initialization.
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
! 
! Ensure types with explicit initialization trigger the gatekeeper and fall back to the old logic.
! CHECK: %[[GLOBAL_INIT_EXP:.*]] = fir.address_of(@_QQ_QMmy_typesTfallbackexplicittype.DerivedInit)
! CHECK: fir.copy %[[GLOBAL_INIT_EXP]] to %[[VAR_EXPLICIT]]