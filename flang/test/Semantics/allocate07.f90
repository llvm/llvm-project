! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

subroutine C936(param_ca_4_assumed, param_ta_4_assumed, param_ca_4_deferred)
! If type-spec appears, the kind type parameter values of each
! allocate-object shall be the same as the corresponding type
! parameter values of the type-spec.

  real(kind=4), allocatable :: x1, x2(:)

  type WithParam(k1, l1)
    integer, kind :: k1=1
    integer, len :: l1=2
  end type

  type, extends(WithParam) :: WithParamExtent(k2, l2)
    integer, kind :: k2
    integer, len :: l2
  end type

  type, extends(WithParamExtent) :: WithParamExtent2(k3, l3)
    integer, kind :: k3 = 8
    integer, len :: l3
  end type

  type(WithParam(4, 2)), allocatable :: param_ta_4_2
  class(WithParam(4, 2)), pointer :: param_ca_4_2

  type(WithParam(4, *)), pointer :: param_ta_4_assumed
  class(WithParam(4, *)), allocatable :: param_ca_4_assumed

  type(WithParam(4, :)), allocatable :: param_ta_4_deferred
  class(WithParam(4, :)), pointer :: param_ca_4_deferred
  class(WithParam), allocatable :: param_defaulted

  type(WithParamExtent2(k1=4, l1=:, k2=5, l2=:, l3=8 )), pointer :: extended2

  class(*), pointer :: whatever

  character(:), allocatable :: deferredChar
  character(2), allocatable :: char2

  ! Nominal test cases
  allocate(real(kind=4):: x1, x2(10))
  allocate(WithParam(4, 2):: param_ta_4_2, param_ca_4_2)
  allocate(WithParamExtent(4, 2, 8, 3):: param_ca_4_2)
  allocate(WithParam(4, *):: param_ta_4_assumed, param_ca_4_assumed)
  allocate(WithParamExtent(4, *, 8, 3):: param_ca_4_assumed)
  allocate(WithParam(4, 2):: param_ta_4_deferred, param_ca_4_deferred)
  allocate(WithParamExtent(4, 2, 8, 3):: param_ca_4_deferred)
  allocate(WithParamExtent2(k1=4, l1=5, k2=5, l2=6, l3=8 ):: extended2)
  allocate(WithParamExtent2(k1=4, l1=2, k2=5, l2=6, k3=5, l3=8 ):: param_ca_4_2)
  allocate(WithParam:: param_defaulted)
  allocate(WithParam(k1=1, l1=2):: param_defaulted)
  allocate(WithParam(k1=1):: param_defaulted)
  allocate(WithParamExtent2(k1=1, l1=2, k2=5, l2=6, k3=5, l3=8 ):: param_defaulted)
  allocate(WithParamExtent2(k1=1, l1=2, k2=5, l2=6, k3=5, l3=8 ):: whatever)
  allocate(character(len=1):: deferredChar)
  allocate(character(len=2):: char2)

  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(real(kind=8):: x1)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(real(kind=8):: x2(10))
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, 2):: param_ta_4_2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, 2):: param_ca_4_2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent(8, 2, 8, 3):: param_ca_4_2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, *):: param_ta_4_assumed)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, *):: param_ca_4_assumed)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent(8, *, 8, 3):: param_ca_4_assumed)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, 2):: param_ta_4_deferred)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(8, 2):: param_ca_4_deferred)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent(8, 2, 8, 3):: param_ca_4_deferred)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent2(k1=5, l1=5, k2=5, l2=6, l3=8 ):: extended2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent2(k1=5, l1=2, k2=5, l2=6, k3=5, l3=8 ):: param_ca_4_2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent2(k1=4, l1=5, k2=5, l2=6, k3=5, l3=8 ):: extended2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam:: param_ca_4_2)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(k1=2, l1=2):: param_defaulted)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParam(k1=2):: param_defaulted)
  !ERROR: Type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec
  allocate(WithParamExtent2(k1=5, l1=2, k2=5, l2=6, k3=5, l3=8 ):: param_defaulted)

  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object has a deferred type parameters
  allocate(deferredChar)
  !ERROR: Character length of allocatable object in ALLOCATE must be the same as the type-spec
  allocate(character(len=1):: char2)

end subroutine
