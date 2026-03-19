! RUN: bbc -emit-fir -enable-precise-init %s -o - | FileCheck %s

! Test for precise component-wise initialization of derived types
! to avoid generating large, fully-initialized global templates (sparse initialization).

module my_types
  type :: InnerType
    integer :: flag = 42
    real, pointer :: p
    real, allocatable :: arr(:)
  end type InnerType

  type :: OuterType
    real, pointer :: q
    integer :: explicit_arr(3) = [10, 20, 30]
    real(8) :: uninit_buffer(80, 100, 100)
    type(InnerType) :: nested
    type(InnerType) :: nested_arr(2)
  end type OuterType
end module my_types

subroutine test_complex_init()
  use my_types
  type(OuterType) :: my_var
  call do_something(my_var)
end subroutine test_complex_init

! ==============================================================================
! FileCheck Assertions
! ==============================================================================

! CHECK-LABEL: func.func @_QPtest_complex_init()

! Ensure we allocate the local variable
! CHECK: %[[MY_VAR:.*]] = fir.alloca !fir.type<_QMmy_typesToutertype{{.*}}>

! ------------------------------------------------------------------------------
! 1. Check pointer 'q' initialization (First component, offset 0, no coordinate_of needed)
! ------------------------------------------------------------------------------
! CHECK: %[[Q_NULL_BOX_ADDR:.*]] = fir.address_of(@_QQ_QMmy_typesToutertype.q.{{(null_box|init)}})
! CHECK: %[[Q_NULL_BOX:.*]] = fir.load %[[Q_NULL_BOX_ADDR]]
! CHECK: fir.store %[[Q_NULL_BOX]] to %{{.*}}

! ------------------------------------------------------------------------------
! 2. Check explicit array initialization
! ------------------------------------------------------------------------------
! CHECK: %[[EXPLICIT_ARR_ADDR:.*]] = fir.coordinate_of %{{.*}}, explicit_arr
! CHECK: %[[EXPLICIT_ARR_INIT:.*]] = fir.address_of(@_QQ_QMmy_typesToutertype.explicit_arr.arr_init)
! CHECK: fir.copy %[[EXPLICIT_ARR_INIT]] to %[[EXPLICIT_ARR_ADDR]]

! ------------------------------------------------------------------------------
! 3. Check scalar derived type 'nested' component-wise precise initialization
! ------------------------------------------------------------------------------
! CHECK: %[[NESTED_ADDR:.*]] = fir.coordinate_of %{{.*}}, nested

! -> Check 'flag' (integer = 42)
! CHECK: %[[FLAG_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], flag
! CHECK: fir.store %c42{{.*}} to %[[FLAG_ADDR]]

! -> Check 'p' (pointer)
! CHECK: %[[P_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], p
! CHECK: %[[P_INIT_ADDR:.*]] = fir.address_of(@_QQ_QMmy_typesTinnertype.p.{{(null_box|init)}})
! CHECK: %[[P_INIT_VAL:.*]] = fir.load %[[P_INIT_ADDR]]
! CHECK: fir.store %[[P_INIT_VAL]] to %[[P_ADDR]]

! -> Check 'arr' (allocatable)
! CHECK: %[[ARR_ADDR:.*]] = fir.coordinate_of %[[NESTED_ADDR]], arr
! CHECK: %[[ARR_INIT_ADDR:.*]] = fir.address_of(@_QQ_QMmy_typesTinnertype.arr.null_box)
! CHECK: %[[ARR_INIT_VAL:.*]] = fir.load %[[ARR_INIT_ADDR]]
! CHECK: fir.store %[[ARR_INIT_VAL]] to %[[ARR_ADDR]]

! ------------------------------------------------------------------------------
! 4. Check derived type array 'nested_arr' initialization (Runtime call)
! ------------------------------------------------------------------------------
! CHECK: %[[NESTED_ARR_ADDR:.*]] = fir.coordinate_of %{{.*}}, nested_arr
! CHECK: %[[NESTED_ARR_BOX:.*]] = fir.embox %[[NESTED_ARR_ADDR]]
! CHECK: %[[NESTED_ARR_PTR:.*]] = fir.convert %[[NESTED_ARR_BOX]]
! CHECK: fir.call @_FortranAInitialize(%[[NESTED_ARR_PTR]], {{.*}})


! ------------------------------------------------------------------------------
! Global Variables Check
! ------------------------------------------------------------------------------

! CRITICAL: Ensure NO full derived type initialization templates are generated!
! We should NOT see fir.global for the entire DerivedInit.
! CHECK-NOT: fir.global internal @_QQ_QMmy_typesToutertype.DerivedInit
! CHECK-NOT: fir.global internal @_QQ_QMmy_typesTinnertype.DerivedInit

! Verify the fine-grained component global constants exist
! CHECK-DAG: fir.global internal @_QQ_QMmy_typesToutertype.explicit_arr.arr_init(dense<[10, 20, 30]> : tensor<3xi32>)
! CHECK-DAG: fir.global internal @_QQ_QMmy_typesToutertype.q.{{(null_box|init)}}
! CHECK-DAG: fir.global internal @_QQ_QMmy_typesTinnertype.p.{{(null_box|init)}}
! CHECK-DAG: fir.global internal @_QQ_QMmy_typesTinnertype.arr.null_box