! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

subroutine C946(param_ca_4_assumed, param_ta_4_assumed, param_ca_4_deferred)
! If source-expr appears, the kind type parameters of each allocate-object shall
! have the same values as the corresponding type parameters of source-expr.

  real(kind=4), allocatable :: x1, x2(:)

  type WithParam(k1, l1)
    integer, kind :: k1=1
    integer, len :: l1=2
    real x
  end type

  type, extends(WithParam) :: WithParamExtent(k2, l2)
    integer, kind :: k2
    integer, len :: l2
  end type

  type, extends(WithParamExtent) :: WithParamExtent2(k3, l3)
    integer, kind :: k3 = 8
    integer, len :: l3
  end type

  real(kind=4) srcx, srcx_array(10)
  real(kind=8) srcx8, srcx8_array(10)
  class(WithParam(4, 2)), allocatable :: src_a_4_2
  type(WithParam(8, 2)) src_a_8_2
  class(WithParam(4, :)), allocatable :: src_a_4_def
  class(WithParam(8, :)), allocatable :: src_a_8_def
  type(WithParamExtent(4, 2, 8, 3)) src_b_4_2_8_3
  class(WithParamExtent(4, :, 8, 3)), allocatable :: src_b_4_def_8_3
  type(WithParamExtent(8, 2, 8, 3)) src_b_8_2_8_3
  class(WithParamExtent(8, :, 8, 3)), allocatable :: src_b_8_def_8_3
  type(WithParamExtent2(k1=4, l1=5, k2=5, l2=6, l3=8 )) src_c_4_5_5_6_8_8
  class(WithParamExtent2(k1=4, l1=2, k2=5, l2=6, k3=5, l3=8)), &
      allocatable :: src_c_4_2_5_6_5_8
  class(WithParamExtent2(k2=5, l2=6, k3=5, l3=8)), &
      allocatable :: src_c_1_2_5_6_5_8
  type(WithParamExtent2(k1=5, l1=5, k2=5, l2=6, l3=8 )) src_c_5_5_5_6_8_8
  type(WithParamExtent2(k1=5, l1=2, k2=5, l2=6, k3=5, l3=8)) src_c_5_2_5_6_5_8


  type(WithParam(4, 2)), allocatable :: param_ta_4_2
  class(WithParam(4, 2)), pointer :: param_ca_4_2

  type(WithParam(4, *)), pointer :: param_ta_4_assumed
  class(WithParam(4, *)), allocatable :: param_ca_4_assumed

  type(WithParam(4, :)), allocatable :: param_ta_4_deferred
  class(WithParam(4, :)), pointer :: param_ca_4_deferred
  class(WithParam), allocatable :: param_defaulted
  integer, allocatable :: integer_default(:)

  type(WithParamExtent2(k1=4, l1=:, k2=5, l2=:, l3=8 )), pointer :: extended2

  class(*), pointer :: whatever

  ! Nominal test cases
  allocate(x1, x2(10), source=srcx)
  allocate(x2(10), source=srcx_array)
  allocate(param_ta_4_2, param_ca_4_2, mold=src_a_4_2)
  allocate(param_ca_4_2, source=src_b_4_2_8_3)
  allocate(param_ta_4_2, param_ca_4_2, mold=src_a_4_def) ! no C935 equivalent for source-expr
  allocate(param_ca_4_2, source=src_b_4_def_8_3) ! no C935 equivalent for source-expr
  allocate(param_ta_4_assumed, param_ca_4_assumed, source=src_a_4_def)
  allocate(param_ca_4_assumed, mold=src_b_4_def_8_3)
  allocate(param_ta_4_assumed, param_ca_4_assumed, source=src_a_4_2) ! no C935 equivalent for source-expr
  allocate(param_ca_4_assumed, mold=src_b_4_2_8_3) ! no C935 equivalent for source-expr
  allocate(param_ta_4_deferred, param_ca_4_deferred, source =src_a_4_2)
  allocate(param_ca_4_deferred, mold=src_b_4_def_8_3)

  allocate(extended2, source=src_c_4_5_5_6_8_8)
  allocate(param_ca_4_2, mold= src_c_4_2_5_6_5_8)
  allocate(param_defaulted, mold=WithParam(5))
  allocate(param_defaulted, source=WithParam(k1=1)(x=5))
  allocate(param_defaulted, mold=src_c_1_2_5_6_5_8)
  allocate(whatever, source=src_c_1_2_5_6_5_8)

  allocate(integer_default, source=[(i,i=0,9)])


  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(x1, source=cos(0._8))
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(x2(10), source=srcx8)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(x2(10), mold=srcx8_array)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ta_4_2, source=src_a_8_2)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_2, mold=src_a_8_2)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ta_4_2, source=src_a_8_def)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_2, source=src_b_8_2_8_3)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_2, mold=src_b_8_def_8_3)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ta_4_assumed, source=src_a_8_def)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ta_4_assumed, mold=src_a_8_2)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_assumed, mold=src_a_8_def)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_assumed, source=src_b_8_2_8_3)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ta_4_deferred, mold=src_a_8_2)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_deferred, source=src_a_8_def)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_deferred, mold=src_b_8_2_8_3)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(extended2, source=src_c_5_5_5_6_8_8)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_2, mold=src_c_5_2_5_6_5_8)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(extended2, source=WithParamExtent2(k1=4, l1=5, k2=5, l2=6, k3=5, l3=8)(x=5))
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_ca_4_2, mold=param_defaulted)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_defaulted, source=param_ca_4_2)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_defaulted, mold=WithParam(k1=2)(x=5))
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(param_defaulted, source=src_c_5_2_5_6_5_8)
  !ERROR: Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression
  allocate(integer_default, source=[(i, integer(8)::i=0,9)])
end subroutine
