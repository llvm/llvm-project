! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test for component-wise initialization of derived types to avoid
! generating large, fully-initialized global templates.

module my_types
  ! Clean internal type: contains only allocatable and pointer components,
  ! without any default initialization.
  type :: InnerClean
    real, pointer :: p
    real, allocatable :: arr(:)
  end type InnerClean

  ! ============================================================================
  ! Test Case 1: Type meeting all criteria for component-wise init (Target)
  ! No arrays of derived types, no default initialization.
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
  ! Test Case 3: Type triggering fallback B (contains default initialization)
  ! ============================================================================
  type :: FallbackDefaultInitType
    real, pointer :: p
    integer :: flag = 999
  end type FallbackDefaultInitType

  ! ============================================================================
  ! Test Case 4: Type triggering fallback C (contains default initialization)
  ! ============================================================================
  type :: ProcPointerType
    real, pointer :: p
    procedure(), pointer, nopass :: pp => null()
  end type ProcPointerType

  ! ============================================================================
  ! Test Case 5: Type with uninitialized procedure pointer.
  ! ============================================================================
  type :: ProcPointerNoInitType
    real, pointer :: p
    procedure(), pointer, nopass :: pp
  end type ProcPointerNoInitType
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
! Test 3: Fallback mechanism for default initialization
! ------------------------------------------------------------------------------
subroutine test_fallback_default_init()
  use my_types
  type(FallbackDefaultInitType) :: var_default_init
  call do_something_default_init(var_default_init)
end subroutine test_fallback_default_init

! CHECK-LABEL: func.func @_QPtest_fallback_default_init()
! CHECK: %[[ALLOCA_DEFAULT:.*]] = fir.alloca !fir.type<_QMmy_typesTfallbackdefaultinittype{{.*}}>
! CHECK: %[[VAR_DEFAULT:.*]] = fir.declare %[[ALLOCA_DEFAULT]]
! CHECK: %[[GLOBAL_INIT_DEFAULT:.*]] = fir.address_of(@_QQ_QMmy_typesTfallbackdefaultinittype.DerivedInit)
! CHECK: fir.copy %[[GLOBAL_INIT_DEFAULT]] to %[[VAR_DEFAULT]]


! ------------------------------------------------------------------------------
! Test 4: Procedure pointers
! Procedure pointers default initialized with '=> null()' trigger the
! default-initialization fallback safely.
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


! ------------------------------------------------------------------------------
! Test 5: Procedure pointers
! Procedure pointers without default initialization should be initialized
! component-wise, without falling back to template copy initialization.
! ------------------------------------------------------------------------------
subroutine test_proc_pointer_no_init()
  use my_types
  type(ProcPointerNoInitType) :: var_proc
  call do_something_proc_no_init(var_proc)
end subroutine test_proc_pointer_no_init

! CHECK-LABEL: func.func @_QPtest_proc_pointer_no_init()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QMmy_typesTprocpointernoinittype{{.*}}>
! CHECK: %[[VAR_DECL:.*]] = fir.declare %[[ALLOCA]] {{.*}}
! CHECK: %[[COORD_P:.*]] = fir.coordinate_of %[[VAR_DECL]], p
! CHECK: %[[NULL_P:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK: %[[BOX_P:.*]] = fir.embox %[[NULL_P]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK: fir.store %[[BOX_P]] to %[[COORD_P]]
! CHECK: %[[COORD_PP:.*]] = fir.coordinate_of %[[VAR_DECL]], pp
! CHECK: %[[NULL_FUNC:.*]] = fir.zero_bits () -> ()
! CHECK: %[[BOX_PP:.*]] = fir.emboxproc %[[NULL_FUNC]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.store %[[BOX_PP]] to %[[COORD_PP]]
! CHECK-NOT: fir.copy
