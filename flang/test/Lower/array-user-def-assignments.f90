! Test lower of elemental user defined assignments
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module defined_assignments
  type t
    integer :: i
  end type
  interface assignment(=)
    elemental subroutine assign_t(a,b)
      import t
      type(t),intent(out) :: a
      type(t),intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_logical_to_real(a,b)
      real, intent(out) :: a
      logical, intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_real_to_logical(a,b)
      logical, intent(out) :: a
      real, intent(in) :: b
    end
  end interface
end module

! CHECK-LABEL: func @_QPtest_derived(
! CHECK-SAME:        %[[arg0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>> {fir.bindc_name = "x"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.region_assign {
! CHECK:       %[[slice:.*]] = hlfir.designate %[[x]]#0 ({{.*}})
! CHECK:       hlfir.yield %[[slice]]
! CHECK:     } to {
! CHECK:       hlfir.yield %[[x]]#0
! CHECK:     } user_defined_assign (%[[rhs:.*]]: !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>) to (%[[lhs:.*]]: !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>) {
! CHECK:       fir.call @_QPassign_t(%[[lhs]], %[[rhs]])
! CHECK:     }

! CHECK-LABEL: func @_QPtest_intrinsic(
! CHECK-SAME:                          %[[arg0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.region_assign {
! CHECK:       hlfir.yield %{{.*}} : !hlfir.expr<100x!fir.logical<4>>
! CHECK:     } to {
! CHECK:       hlfir.yield %[[x]]#0
! CHECK:     } user_defined_assign (%[[rhs:.*]]: !fir.logical<4>) to (%[[lhs:.*]]: !fir.ref<f32>) {
! CHECK:       %[[assoc:.*]]:3 = hlfir.associate %[[rhs]]
! CHECK:       fir.call @_QPassign_logical_to_real(%[[lhs]], %[[assoc]]#0)
! CHECK:     }

! CHECK-LABEL: func @_QPtest_intrinsic_2(
! CHECK-SAME:                            %[[arg0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.region_assign {
! CHECK:       hlfir.yield %[[y]]#0
! CHECK:     } to {
! CHECK:       hlfir.yield %[[x]]#0
! CHECK:     } user_defined_assign (%[[rhs:.*]]: !fir.ref<f32>) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:       fir.call @_QPassign_real_to_logical(%[[lhs]], %[[rhs]])
! CHECK:     }

! CHECK-LABEL: func @_QPfrom_char(
! CHECK-SAME:                     %[[arg0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[c:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     %[[i:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.region_assign {
! CHECK:       hlfir.yield %[[c]]#0
! CHECK:     } to {
! CHECK:       hlfir.yield %[[i]]#0
! CHECK:     } user_defined_assign (%[[rhs:.*]]: !fir.boxchar<1>) to (%[[lhs:.*]]: !fir.ref<i32>) {
! CHECK:       fir.call @_QPsfrom_char(%[[lhs]], %[[rhs]])
! CHECK:     }

! CHECK-LABEL: func @_QPto_char(
! CHECK-SAME:                   %[[arg0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[c:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     %[[i:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.region_assign {
! CHECK:       hlfir.yield %[[i]]#0
! CHECK:     } to {
! CHECK:       hlfir.yield %[[c]]#0
! CHECK:     } user_defined_assign (%[[rhs:.*]]: !fir.ref<i32>) to (%[[lhs:.*]]: !fir.boxchar<1>) {
! CHECK:       fir.call @_QPsto_char(%[[lhs]], %[[rhs]])
! CHECK:     }

subroutine test_derived(x)
  use defined_assignments
  type(t) :: x(100)
  x = x(100:1:-1)
end subroutine

subroutine test_intrinsic(x)
  use defined_assignments
  real :: x(100)
  x = x(100:1:-1) .lt. 0.
end subroutine

subroutine test_intrinsic_2(x, y)
  use defined_assignments
  logical :: x(100)
  real :: y(100)
  x = y
end subroutine

subroutine from_char(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  i = c
end subroutine

subroutine to_char(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  c = i
end subroutine

! -----------------------------------------------------------------------------
!     Test user defined assignments inside FORALL and WHERE
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_in_forall_1(
! CHECK-SAME:                            %[[arg0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : f32
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: f32) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[rhs]]
! CHECK:         fir.call @_QPassign_real_to_logical(%[[lhs]], %[[assoc]]#0)
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPtest_in_forall_2(
! CHECK-SAME:                            %[[arg0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : i1
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.ref<f32>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: i1) to (%[[lhs:.*]]: !fir.ref<f32>) {
! CHECK:         %[[cast:.*]] = fir.convert %[[rhs]]
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[cast]]
! CHECK:         fir.call @_QPassign_logical_to_real(%[[lhs]], %[[assoc]]#0)
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPtest_intrinsic_where_1(
! CHECK-SAME:             %[[arg0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}, %[[arg2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:     %[[l:.*]]:2 = hlfir.declare %[[arg2]]
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.where {
! CHECK:       hlfir.yield %[[l]]#0
! CHECK:     } do {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield %[[y]]#0
! CHECK:       } to {
! CHECK:         hlfir.yield %[[x]]#0
! CHECK:       } user_defined_assign (%[[rhs:.*]]: !fir.ref<f32>) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:         fir.call @_QPassign_real_to_logical(%[[lhs]], %[[rhs]])
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPtest_intrinsic_where_2(
! CHECK-SAME:           %[[arg0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}, %[[arg2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:     %[[l:.*]]:2 = hlfir.declare %[[arg2]]
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.where {
! CHECK:       hlfir.yield %[[l]]#0
! CHECK:     } do {
! CHECK:       hlfir.region_assign {
! CHECK:         %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
! CHECK:           hlfir.yield_element
! CHECK:         }
! CHECK:         hlfir.yield %[[res]] : !hlfir.expr<10x!fir.logical<4>> cleanup {
! CHECK:           hlfir.destroy %[[res]]
! CHECK:         }
! CHECK:       } to {
! CHECK:         hlfir.yield %[[y]]#0
! CHECK:       } user_defined_assign (%[[rhs:.*]]: !fir.logical<4>) to (%[[lhs:.*]]: !fir.ref<f32>) {
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[rhs]]
! CHECK:         fir.call @_QPassign_logical_to_real(%[[lhs]], %[[assoc]]#0)
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPtest_scalar_func_but_not_elemental(
! CHECK-SAME:        %[[arg0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : i32
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: i32) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[rhs]]
! CHECK:         fir.call @_QPassign_integer_to_logical(%[[lhs]], %[[assoc]]#0)
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPtest_in_forall_with_cleanup(
! CHECK-SAME:       %[[arg0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : f32 cleanup {
! CHECK:         }
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: f32) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:         %[[assoc:.*]]:3 = hlfir.associate %[[rhs]]
! CHECK:         fir.call @_QPassign_real_to_logical(%[[lhs]], %[[assoc]]#0)
! CHECK:       }
! CHECK:     }

subroutine test_in_forall_1(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = y(i)
end subroutine

subroutine test_in_forall_2(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) y(i) = y(i).lt.0.
end subroutine

subroutine test_intrinsic_where_1(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) x = y
end subroutine

subroutine test_intrinsic_where_2(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) y = y.lt.0.
end subroutine

subroutine test_scalar_func_but_not_elemental(x, y)
  interface assignment(=)
    ! scalar, but not elemental
    elemental subroutine assign_integer_to_logical(a,b)
      logical, intent(out) :: a
      integer, intent(in) :: b
    end
  end interface
  logical :: x(100)
  integer :: y(100)
  ! Scalar assignment in forall should be treated just like elemental
  ! functions.
  forall(i=1:10) x(i) = y(i)
end subroutine

subroutine test_in_forall_with_cleanup(x, y)
  use defined_assignments
  interface
    pure function returns_alloc(i)
      integer, intent(in) :: i
      real, allocatable :: returns_alloc
    end function
  end interface
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = returns_alloc(i)
end subroutine

! CHECK-LABEL: func @_QPtest_forall_array(
! CHECK-SAME:        %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %[[arg1:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[x:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     %[[y:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: !fir.ref<f32>) to (%[[lhs:.*]]: !fir.ref<!fir.logical<4>>) {
! CHECK:         fir.call @_QPassign_real_to_logical(%[[lhs]], %[[rhs]])
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPfrom_char_forall_array(
! CHECK-SAME:       %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[c:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     %[[i:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?xi32>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: !fir.boxchar<1>) to (%[[lhs:.*]]: !fir.ref<i32>) {
! CHECK:         fir.call @_QPsfrom_char(%[[lhs]], %[[rhs]])
! CHECK:       }
! CHECK:     }

! CHECK-LABEL: func @_QPto_char_forall_array(
! CHECK-SAME:        %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[c:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK:     %[[i:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:     hlfir.forall {{.*}} {
! CHECK:       hlfir.region_assign {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?xi32>>
! CHECK:       } to {
! CHECK:         hlfir.yield {{.*}} : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:       } user_defined_assign (%[[rhs:.*]]: !fir.ref<i32>) to (%[[lhs:.*]]: !fir.boxchar<1>) {
! CHECK:         fir.call @_QPsto_char(%[[lhs]], %[[rhs]])
! CHECK:       }
! CHECK:     }

subroutine test_forall_array(x, y)
  use defined_assignments
  logical :: x(:, :)
  real :: y(:, :)
  forall (i=1:10) x(i, :) = y(i, :)
end subroutine

subroutine from_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) i(j, :) = c(j, :)
end subroutine

subroutine to_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) c(j, :) = i(j, :)
end subroutine

! TODO: test array user defined assignment inside FORALL.
subroutine test_todo(x, y)
  interface assignment(=)
    ! User assignment is not elemental, it takes array arguments.
    pure subroutine assign_array(a,b)
      logical, intent(out) :: a(:)
      integer, intent(in) :: b(:)
    end
  end interface
  logical :: x(10, 10)
  integer :: y(10, 10)
!  forall(i=1:10) x(i, :) = y(i, :)
end subroutine