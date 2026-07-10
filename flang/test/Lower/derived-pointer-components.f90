! Test lowering of pointer components
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module pcomp
  implicit none
  type t
    real :: x
    integer :: i
  end type
  interface
    subroutine takes_real_scalar(x)
      real :: x
    end subroutine
    subroutine takes_char_scalar(x)
      character(*) :: x
    end subroutine
    subroutine takes_derived_scalar(x)
      import t
      type(t) :: x
    end subroutine
    subroutine takes_real_array(x)
      real :: x(:)
    end subroutine
    subroutine takes_char_array(x)
      character(*) :: x(:)
    end subroutine
    subroutine takes_derived_array(x)
      import t
      type(t) :: x(:)
    end subroutine
    subroutine takes_real_scalar_pointer(x)
      real, pointer :: x
    end subroutine
    subroutine takes_real_array_pointer(x)
      real, pointer :: x(:)
    end subroutine
    subroutine takes_logical(x)
      logical :: x
    end subroutine
  end interface

  type real_p0
    real, pointer :: p
  end type
  type real_p1
    real, pointer :: p(:)
  end type
  type cst_char_p0
    character(10), pointer :: p
  end type
  type cst_char_p1
    character(10), pointer :: p(:)
  end type
  type def_char_p0
    character(:), pointer :: p
  end type
  type def_char_p1
    character(:), pointer :: p(:)
  end type
  type derived_p0
    type(t), pointer :: p
  end type
  type derived_p1
    type(t), pointer :: p(:)
  end type

  real, target :: real_target, real_array_target(100)
  character(10), target :: char_target, char_array_target(100)

contains

! -----------------------------------------------------------------------------
!            Test pointer component references
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPref_scalar_real_p(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>{{.*}}, %[[ARG1:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>>{{.*}}, %[[ARG3:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine ref_scalar_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[VAL:.*]] = fir.convert %[[ADDR]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[VAL]])
  call takes_real_scalar(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[VAL:.*]] = fir.convert %[[ADDR]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[VAL]])
  call takes_real_scalar(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[ELT]])
  call takes_real_scalar(p1_0%p(7))

  ! CHECK: %[[ELT_ARR:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT_ARR]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[ELT]])
  call takes_real_scalar(p1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPref_array_real_p(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>{{.*}}, %[[ARG1:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine ref_array_real_p(p1_0, p1_1)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG1]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[SLICE:.*]] = hlfir.designate %[[LOAD]] (%c20{{.*}}:%c50{{.*}}:%c2{{.*}})
  ! CHECK: %[[BOX:.*]] = fir.convert %[[SLICE]]
  ! CHECK: fir.call @_QPtakes_real_array(%[[BOX]])
  call takes_real_array(p1_0%p(20:50:2))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[SLICE:.*]] = hlfir.designate %[[LOAD]] (%c20{{.*}}:%c50{{.*}}:%c2{{.*}})
  ! CHECK: %[[BOX:.*]] = fir.convert %[[SLICE]]
  ! CHECK: fir.call @_QPtakes_real_array(%[[BOX]])
  call takes_real_array(p1_1(5)%p(20:50:2))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPassign_scalar_real_p
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine assign_scalar_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: hlfir.assign %{{.*}} to %[[ADDR]]
  p0_0%p = 1.

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: hlfir.assign %{{.*}} to %[[ADDR]]
  p0_1(5)%p = 2.

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: hlfir.assign %{{.*}} to %[[ADDR]]
  p1_0%p(7) = 3.

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: hlfir.assign %{{.*}} to %[[ADDR]]
  p1_1(5)%p(7) = 4.
end subroutine

! CHECK-LABEL: func.func @_QMpcompPref_scalar_cst_char_p
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine ref_scalar_cst_char_p(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(cst_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[VAL:.*]] = fir.convert %[[ADDR]]
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[VAL]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[VAL:.*]] = fir.convert %[[ADDR]]
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[VAL]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[ADDR]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p1_0%p(7))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[ADDR]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPref_scalar_def_char_p
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine ref_scalar_def_char_p(p0_0, p1_0, p0_1, p1_1)
  type(def_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[LOAD2:.*]] = fir.load %[[P]]
  ! CHECK: %[[LEN:.*]] = fir.box_elesize %[[LOAD2]]
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[ADDR]], %[[LEN]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[LOAD2:.*]] = fir.load %[[P]]
  ! CHECK: %[[LEN:.*]] = fir.box_elesize %[[LOAD2]]
  ! CHECK: %[[BOXCHAR:.*]] = fir.emboxchar %[[ADDR]], %[[LEN]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[LEN:.*]] = fir.box_elesize %[[LOAD]]
  ! CHECK: %[[BOXCHAR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}}) typeparams %[[LEN]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p1_0%p(7))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[LEN:.*]] = fir.box_elesize %[[LOAD]]
  ! CHECK: %[[BOXCHAR:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}}) typeparams %[[LEN]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[BOXCHAR]])
  call takes_char_scalar(p1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPref_scalar_derived
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine ref_scalar_derived(p0_0, p1_0, p0_1, p1_1)
  type(derived_p0) :: p0_0, p0_1(100)
  type(derived_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[X:.*]] = hlfir.designate %[[ADDR]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[X]])
  call takes_real_scalar(p0_0%p%x)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: %[[X:.*]] = hlfir.designate %[[ADDR]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[X]])
  call takes_real_scalar(p0_1(5)%p%x)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: %[[X:.*]] = hlfir.designate %[[ELT]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[X]])
  call takes_real_scalar(p1_0%p(7)%x)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ELT2:.*]] = hlfir.designate %[[LOAD]] (%c7{{.*}})
  ! CHECK: %[[X:.*]] = hlfir.designate %[[ELT2]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[X]])
  call takes_real_scalar(p1_1(5)%p(7)%x)
end subroutine

! -----------------------------------------------------------------------------
!            Test passing pointer component references as pointers
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPpass_real_p
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine pass_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[P]])
  call takes_real_scalar_pointer(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[P]])
  call takes_real_scalar_pointer(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[P]])
  call takes_real_array_pointer(p1_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[P]])
  call takes_real_array_pointer(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test usage in intrinsics where pointer aspect matters
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPassociated_p
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine associated_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: fir.convert %[[ADDR]] : (!fir.ptr<f32>) -> i64
  call takes_logical(associated(p0_0%p))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: fir.convert %[[ADDR]] : (!fir.ptr<f32>) -> i64
  call takes_logical(associated(p0_1(5)%p))

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> i64
  call takes_logical(associated(p1_0%p))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[LOAD:.*]] = fir.load %[[P]]
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]]
  ! CHECK: fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> i64
  call takes_logical(associated(p1_1(5)%p))
end subroutine

! -----------------------------------------------------------------------------
!            Test pointer assignment of components
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPpassoc_real
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine passoc_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p0_0%p => real_target

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p0_1(5)%p => real_target

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p1_0%p => real_array_target

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p1_1(5)%p => real_array_target
end subroutine

! CHECK-LABEL: func.func @_QMpcompPpassoc_char
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine passoc_char(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p0_0%p => char_target

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p0_1(5)%p => char_target

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p1_0%p => char_array_target

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.store {{.*}} to %[[P]]
  p1_1(5)%p => char_array_target
end subroutine

! -----------------------------------------------------------------------------
!            Test nullify of components
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPnullify_test
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine nullify_test(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: %[[NULL:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[NULL]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[BOX]] to %[[P]]
  nullify(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[NULL:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[NULL]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[BOX]] to %[[P]]
  nullify(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: %[[NULL:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[NULL]]
  ! CHECK: fir.store %[[BOX]] to %[[P]]
  nullify(p1_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: %[[NULL:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[NULL]]
  ! CHECK: fir.store %[[BOX]] to %[[P]]
  nullify(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test allocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPallocate_real
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine allocate_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p1_0%p(100))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPallocate_cst_char
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine allocate_cst_char(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(cst_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p1_0%p(100))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(p1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func.func @_QMpcompPallocate_def_char
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine allocate_def_char(p0_0, p1_0, p0_1, p1_1)
  type(def_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(character(18)::p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(character(18)::p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(character(18)::p1_0%p(100))

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerSetBounds
  ! CHECK: fir.call @_FortranAPointerAllocate
  allocate(character(18)::p1_1(5)%p(100))
end subroutine

! -----------------------------------------------------------------------------
!            Test deallocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPdeallocate_real
! CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
subroutine deallocate_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[P0_0_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[P0_1_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[P1_0_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[P1_1_DECL:.*]]:2 = hlfir.declare %[[ARG3]]

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P0_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerDeallocate
  deallocate(p0_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P0_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerDeallocate
  deallocate(p0_1(5)%p)

  ! CHECK: %[[P:.*]] = hlfir.designate %[[P1_0_DECL]]#0{"p"}
  ! CHECK: fir.call @_FortranAPointerDeallocate
  deallocate(p1_0%p)

  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[P1_1_DECL]]#0 (%c5{{.*}})
  ! CHECK: %[[P:.*]] = hlfir.designate %[[ELT]]{"p"}
  ! CHECK: fir.call @_FortranAPointerDeallocate
  deallocate(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test a very long component
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPvery_long
! CHECK-SAME: (%[[X:.*]]: {{.*}})
subroutine very_long(x)
  type t0
    real :: f
  end type
  type t1
    type(t0), allocatable :: e(:)
  end type
  type t2
    type(t1) :: d(10)
  end type
  type t3
    type(t2) :: c
  end type
  type t4
    type(t3), pointer :: b
  end type
  type t5
    type(t4) :: a
  end type
  type(t5) :: x(:, :, :, :, :)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: %[[X_ELT:.*]] = hlfir.designate %[[X_DECL]]#0 (%c1{{.*}}, %c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}})
  ! CHECK: %[[A:.*]] = hlfir.designate %[[X_ELT]]{"a"}
  ! CHECK: %[[B:.*]] = hlfir.designate %[[A]]{"b"}
  ! CHECK: %[[B_LOAD:.*]] = fir.load %[[B]]
  ! CHECK: %[[B_ADDR:.*]] = fir.box_addr %[[B_LOAD]]
  ! CHECK: %[[C:.*]] = hlfir.designate %[[B_ADDR]]{"c"}
  ! CHECK: %[[D_ELT:.*]] = hlfir.designate %[[C]]{"d"} <%{{.*}}> (%c6{{.*}})
  ! CHECK: %[[E:.*]] = hlfir.designate %[[D_ELT]]{"e"}
  ! CHECK: %[[E_LOAD:.*]] = fir.load %[[E]]
  ! CHECK: %[[E_ELT:.*]] = hlfir.designate %[[E_LOAD]] (%c7{{.*}})
  ! CHECK: %[[F:.*]] = hlfir.designate %[[E_ELT]]{"f"}
  ! CHECK: fir.load %[[F]]
  print *, x(1,2,3,4,5)%a%b%c%d(6)%e(7)%f
end subroutine

! -----------------------------------------------------------------------------
!            Test a recursive derived type reference
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QMpcompPtest_recursive
! CHECK-SAME: (%[[X:.*]]: {{.*}})
subroutine test_recursive(x)
  type t
    integer :: i
    type(t), pointer :: next
  end type
  type(t) :: x
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: %[[NEXT:.*]] = hlfir.designate %[[X_DECL]]#0{"next"}
  ! CHECK: %[[NEXT_LOAD:.*]] = fir.load %[[NEXT]]
  ! CHECK: %[[NEXT_ADDR:.*]] = fir.box_addr %[[NEXT_LOAD]]
  ! CHECK: %[[NEXT2:.*]] = hlfir.designate %[[NEXT_ADDR]]{"next"}
  ! CHECK: %[[NEXT2_LOAD:.*]] = fir.load %[[NEXT2]]
  ! CHECK: %[[NEXT2_ADDR:.*]] = fir.box_addr %[[NEXT2_LOAD]]
  ! CHECK: %[[NEXT3:.*]] = hlfir.designate %[[NEXT2_ADDR]]{"next"}
  ! CHECK: %[[NEXT3_LOAD:.*]] = fir.load %[[NEXT3]]
  ! CHECK: %[[NEXT3_ADDR:.*]] = fir.box_addr %[[NEXT3_LOAD]]
  ! CHECK: %[[I:.*]] = hlfir.designate %[[NEXT3_ADDR]]{"i"}
  ! CHECK: fir.load %[[I]]
  print *, x%next%next%next%i
end subroutine

end module

! -----------------------------------------------------------------------------
!            Test initial data target
! -----------------------------------------------------------------------------

module pinit
  use pcomp
  ! CHECK-LABEL: fir.global {{.*}}@_QMpinitEarp0
    ! CHECK-DAG: %[[undef:.*]] = fir.undefined
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEreal_target)
    ! CHECK: %[[decl:.*]]:2 = hlfir.declare %[[target]]
    ! CHECK: %[[box:.*]] = fir.embox %[[decl]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
    ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(real_p0) :: arp0 = real_p0(real_target)

! CHECK-LABEL: fir.global @_QMpinitEbrp1 : !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}> {
! CHECK:         %[[VAL_0:.*]] = fir.undefined !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QMpcompEreal_array_target) : !fir.ref<!fir.array<100xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_18]])
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK-DAG:     %[[VAL_9:.*]] = arith.constant 5 : index
! CHECK-DAG:     %[[VAL_11:.*]] = arith.constant 50 : index
! CHECK:         %[[VAL_19:.*]] = hlfir.designate %[[VAL_DECL]]#0 (%[[VAL_7]]:%[[VAL_11]]:%[[VAL_9]])
! CHECK:         %[[VAL_21:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.array<9xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         %[[VAL_22:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_21]], ["p", !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>] : (!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         fir.has_value %[[VAL_22]] : !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:       }
  type(real_p1) :: brp1 = real_p1(real_array_target(10:50:5))

  ! CHECK-LABEL: fir.global {{.*}}@_QMpinitEccp0
    ! CHECK-DAG: %[[undef:.*]] = fir.undefined
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEchar_target)
    ! CHECK: %[[decl:.*]]:2 = hlfir.declare %[[target]]
    ! CHECK: %[[box:.*]] = fir.embox %[[decl]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
    ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTcst_char_p0{p:!fir.box<!fir.ptr<!fir.char<1,10>>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(cst_char_p0) :: ccp0 = cst_char_p0(char_target)

  ! CHECK-LABEL: fir.global {{.*}}@_QMpinitEdcp1
    ! CHECK-DAG: %[[undef:.*]] = fir.undefined
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEchar_array_target)
    ! CHECK-DAG: %[[shape:.*]] = fir.shape %c100{{.*}}
    ! CHECK: %[[decl:.*]]:2 = hlfir.declare %[[target]](%[[shape]])
    ! CHECK-DAG: %[[shape2:.*]] = fir.shape %c100{{.*}}
    ! CHECK-DAG: %[[box:.*]] = fir.embox %[[decl]]#0(%[[shape2]]) : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<100x!fir.char<1,10>>>
    ! CHECK-DAG: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<100x!fir.char<1,10>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTdef_char_p1{p:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(def_char_p1) :: dcp1 = def_char_p1(char_array_target)
end module
