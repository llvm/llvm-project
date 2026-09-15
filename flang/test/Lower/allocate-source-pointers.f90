! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Test lowering of pointers for allocate statements with source.

! CHECK-LABEL: func.func @_QPtest_pointer_scalar(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f32>
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[ADDR_X1:.*]] = fir.address_of(@_QFtest_pointer_scalarEx1)
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[ADDR_X1]]
! CHECK: %[[ADDR_X2:.*]] = fir.address_of(@_QFtest_pointer_scalarEx2)
! CHECK: %[[DECL_X2:.*]]:2 = hlfir.declare %[[ADDR_X2]]
! CHECK: %[[FALSE:.*]] = arith.constant false
! CHECK: %[[ABSENT:.*]] = fir.absent !fir.box<none>
! CHECK: %[[BOX_A:.*]] = fir.embox %[[DECL_A]]#0
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %[[FALSE]], %[[ABSENT]], %{{.*}}, %{{.*}})
! CHECK: %[[BOX_X2:.*]] = fir.convert %[[DECL_X2]]#0
! CHECK: %[[BOX_SOURCE_2:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X2]], %[[BOX_SOURCE_2]], %[[FALSE]], %[[ABSENT]], %{{.*}}, %{{.*}})

subroutine test_pointer_scalar(a)
  real, save, pointer :: x1, x2
  real :: a

  allocate(x1, x2, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_2d_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[ARG1:.*]]: !fir.ref<!fir.array<?x?xi32>>
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[X1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {{{.*}}uniq_name = "_QFtest_pointer_2d_arrayEx1"}
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[X1]]
! CHECK: %[[X2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {{{.*}}uniq_name = "_QFtest_pointer_2d_arrayEx2"}
! CHECK: %[[DECL_X2:.*]]:2 = hlfir.declare %[[X2]]
! CHECK: %[[X3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {{{.*}}uniq_name = "_QFtest_pointer_2d_arrayEx3"}
! CHECK: %[[DECL_X3:.*]]:2 = hlfir.declare %[[X3]]
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[BOX_A:.*]] = fir.embox %[[DECL_A]]#1
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X2:.*]] = fir.convert %[[DECL_X2]]#0
! CHECK: %[[BOX_SOURCE_2:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X2]], %[[BOX_SOURCE_2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
! CHECK: hlfir.designate %[[DECL_A]]#0
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X3:.*]] = fir.convert %[[DECL_X3]]#0
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_2d_array(n, a)
  integer, pointer :: x1(:,:), x2(:,:), x3(:,:)
  integer :: n, sss, a(n, n)

  allocate(x1, x2, source = a)
  allocate(x3, source = a(1:3:2, 2:3), stat=sss)
end

! CHECK-LABEL: func.func @_QPtest_pointer_with_shapespec(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>{{.*}}%[[ARG1:.*]]: !fir.ref<!fir.array<?xi32>>{{.*}}%[[ARG2:.*]]: !fir.ref<i32>
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[DECL_M:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK: %[[X1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[X1]]
! CHECK: %[[X2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[DECL_X2:.*]]:2 = hlfir.declare %[[X2]]
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[BOX_A:.*]] = fir.embox %[[DECL_A]]#1
! CHECK: fir.call @_FortranAPointerSetBounds(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X2:.*]] = fir.convert %[[DECL_X2]]#0
! CHECK: %[[BOX_SOURCE_2:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X2]], %[[BOX_SOURCE_2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_with_shapespec(n, a, m)
  integer, pointer :: x1(:), x2(:)
  integer :: n, m, a(n)

  allocate(x1(2:m), x2(n), source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_from_const(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[X1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[X1]]
! CHECK: %[[CONST_ADDR:.*]] = fir.address_of(@_QQro.5xi4.0)
! CHECK: %[[DECL_CONST:.*]]:2 = hlfir.declare %[[CONST_ADDR]]
! CHECK: %[[BOX_CONST:.*]] = fir.embox %[[DECL_CONST]]#0
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_CONST]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_from_const(n, a)
  integer, pointer :: x1(:)
  integer :: n, a(n)

  allocate(x1, source = [1, 2, 3, 4, 5])
end

! CHECK-LABEL: func.func @_QPtest_pointer_chararray(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[ARG1:.*]]: !fir.boxchar<1>
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[X1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[X1]]
! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[ARG1]]
! CHECK: %[[CONV:.*]] = fir.convert %[[UNBOX]]#0
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[CONV]]
! CHECK: %[[BOX_A:.*]] = fir.embox %[[DECL_A]]#1
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_chararray(n, a)
  character(4), pointer :: x1(:)
  integer :: n
  character(*) :: a(n)

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_char(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[ARG1:.*]]: !fir.boxchar<1>
! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[ARG1]]
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[UNBOX]]#0
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[X1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[DECL_X1:.*]]:2 = hlfir.declare %[[X1]]
! CHECK: %[[BOX_A:.*]] = fir.embox %[[DECL_A]]#1
! CHECK: fir.call @_FortranAPointerNullifyCharacter
! CHECK: %[[BOX_X1:.*]] = fir.convert %[[DECL_X1]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[BOX_A]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_X1]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_char(n, a)
  character(:), pointer :: x1
  integer :: n
  character(*) :: a

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_derived_type(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK: %[[DECL_Y:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[Z:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>
! CHECK: %[[DECL_Z:.*]]:2 = hlfir.declare %[[Z]]
! CHECK: %[[LOAD_Y:.*]] = fir.load %[[DECL_Y]]#0
! CHECK: fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_Z:.*]] = fir.convert %[[DECL_Z]]#0
! CHECK: %[[BOX_SOURCE:.*]] = fir.convert %[[LOAD_Y]]
! CHECK: fir.call @_FortranAPointerAllocateSource(%[[BOX_Z]], %[[BOX_SOURCE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})

subroutine test_pointer_derived_type(y)
  type t
    integer, pointer :: x(:)
  end type
  type(t), pointer :: z(:), y(:)

  allocate(z, source=y)
end
