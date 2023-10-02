! Test lowering of pointer components
! RUN: bbc -emit-fir %s -o - | FileCheck %s

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

! CHECK-LABEL: func @_QMpcompPref_scalar_real_p(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine ref_scalar_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg0]], %[[fld]] : (!fir.ref<!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(p0_0%p)

  ! CHECK: %[[p0_1_coor:.*]] = fir.coordinate_of %[[arg2]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>>, i64) -> !fir.ref<!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg1]], %[[fld]] : (!fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[load]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[load]], %[[index]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(p1_0%p(7))

  ! CHECK: %[[p1_1_coor:.*]] = fir.coordinate_of %[[arg3]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[load]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[load]], %[[index]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(p1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func @_QMpcompPref_array_real_p(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_5]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_7:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = fir.shift %[[VAL_6]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_15:.*]] = fir.rebox %[[VAL_4]](%[[VAL_13]]) {{\[}}%[[VAL_14]]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[VAL_15_NEW:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[VAL_15_NEW]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         %[[VAL_16:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_17]] : i64
! CHECK:         %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_18]] : (!fir.ref<!fir.array<100x!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_20:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_20]] : (!fir.ref<!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_22]], %[[VAL_23]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_25:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:         %[[VAL_29:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:         %[[VAL_31:.*]] = fir.shift %[[VAL_24]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_32:.*]] = fir.slice %[[VAL_26]], %[[VAL_30]], %[[VAL_28]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_33:.*]] = fir.rebox %[[VAL_22]](%[[VAL_31]]) {{\[}}%[[VAL_32]]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[VAL_33_NEW:.*]] = fir.convert %[[VAL_33]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[VAL_33_NEW]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         return
! CHECK:       }


subroutine ref_array_real_p(p1_0, p1_1)
  type(real_p1) :: p1_0, p1_1(100)
  call takes_real_array(p1_0%p(20:50:2))
  call takes_real_array(p1_1(5)%p(20:50:2))
end subroutine

! CHECK-LABEL: func @_QMpcompPassign_scalar_real
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine assign_scalar_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: fir.store {{.*}} to %[[addr]]
  p0_0%p = 1.

  ! CHECK: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: fir.store {{.*}} to %[[addr]]
  p0_1(5)%p = 2.

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], {{.*}}
  ! CHECK: fir.store {{.*}} to %[[addr]]
  p1_0%p(7) = 3.

  ! CHECK: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], {{.*}}
  ! CHECK: fir.store {{.*}} to %[[addr]]
  p1_1(5)%p(7) = 4.
end subroutine

! CHECK-LABEL: func @_QMpcompPref_scalar_cst_char_p
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine ref_scalar_cst_char_p(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(cst_char_p1) :: p1_0, p1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMpcompPref_scalar_def_char_p
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine ref_scalar_def_char_p(p0_0, p1_0, p0_1, p1_1)
  type(def_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK-DAG: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK-DAG: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK-DAG: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK-DAG: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK-DAG: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK-DAG: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(p1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMpcompPref_scalar_derived
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine ref_scalar_derived(p0_0, p1_0, p0_1, p1_1)
  type(derived_p0) :: p0_0, p0_1(100)
  type(derived_p1) :: p1_0, p1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(p0_0%p%x)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(p0_1(5)%p%x)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[elem]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(p1_0%p(7)%x)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[elem]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(p1_1(5)%p(7)%x)

end subroutine

! -----------------------------------------------------------------------------
!            Test passing pointer component references as pointers
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPpass_real_p
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine pass_real_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(p1_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test usage in intrinsics where pointer aspect matters
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPassociated_p
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine associated_p(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(associated(p0_0%p))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(associated(p0_1(5)%p))

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(associated(p1_0%p))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(associated(p1_1(5)%p))
end subroutine

! -----------------------------------------------------------------------------
!            Test pointer assignment of components
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPpassoc_real
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine passoc_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p0_0%p => real_target

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p0_1(5)%p => real_target

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p1_0%p => real_array_target

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p1_1(5)%p => real_array_target
end subroutine

! CHECK-LABEL: func @_QMpcompPpassoc_char
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine passoc_char(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p0_0%p => char_target

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p0_1(5)%p => char_target

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p1_0%p => char_array_target

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  p1_1(5)%p => char_array_target
end subroutine

! -----------------------------------------------------------------------------
!            Test nullify of components
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPnullify_test
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine nullify_test(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  nullify(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  nullify(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  nullify(p1_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  nullify(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test allocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPallocate_real
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine allocate_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMpcompPallocate_cst_char
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine allocate_cst_char(p0_0, p1_0, p0_1, p1_1)
  type(cst_char_p0) :: p0_0, p0_1(100)
  type(cst_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(p1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMpcompPallocate_def_char
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine allocate_def_char(p0_0, p1_0, p0_1, p1_1)
  type(def_char_p0) :: p0_0, p0_1(100)
  type(def_char_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::p1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::p1_1(5)%p(100))
end subroutine

! -----------------------------------------------------------------------------
!            Test deallocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPdeallocate_real
! CHECK-SAME: (%[[p0_0:.*]]: {{.*}}, %[[p1_0:.*]]: {{.*}}, %[[p0_1:.*]]: {{.*}}, %[[p1_1:.*]]: {{.*}})
subroutine deallocate_real(p0_0, p1_0, p0_1, p1_1)
  type(real_p0) :: p0_0, p0_1(100)
  type(real_p1) :: p1_0, p1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(p0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(p0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[p1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(p1_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[p1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(p1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test a very long component
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMpcompPvery_long
! CHECK-SAME: (%[[x:.*]]: {{.*}})
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

  ! CHECK: %[[coor0:.*]] = fir.coordinate_of %[[x]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.}}
  ! CHECK-DAG: %[[flda:.*]] = fir.field_index a
  ! CHECK-DAG: %[[fldb:.*]] = fir.field_index b
  ! CHECK: %[[coor1:.*]] = fir.coordinate_of %[[coor0]], %[[flda]], %[[fldb]]
  ! CHECK: %[[b_box:.*]] = fir.load %[[coor1]]
  ! CHECK-DAG: %[[fldc:.*]] = fir.field_index c
  ! CHECK-DAG: %[[fldd:.*]] = fir.field_index d
  ! CHECK: %[[coor2:.*]] = fir.coordinate_of %[[b_box]], %[[fldc]], %[[fldd]]
  ! CHECK: %[[index:.*]] = arith.subi %c6{{.*}}, %c1{{.*}} : i64
  ! CHECK: %[[coor3:.*]] = fir.coordinate_of %[[coor2]], %[[index]]
  ! CHECK: %[[flde:.*]] = fir.field_index e
  ! CHECK: %[[coor4:.*]] = fir.coordinate_of %[[coor3]], %[[flde]]
  ! CHECK: %[[e_box:.*]] = fir.load %[[coor4]]
  ! CHECK: %[[edims:.*]]:3 = fir.box_dims %[[e_box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[edims]]#0 : (index) -> i64
  ! CHECK: %[[index2:.*]] = arith.subi %c7{{.*}}, %[[lb]]
  ! CHECK: %[[coor5:.*]] = fir.coordinate_of %[[e_box]], %[[index2]]
  ! CHECK: %[[fldf:.*]] = fir.field_index f
  ! CHECK: %[[coor6:.*]] = fir.coordinate_of %[[coor5]], %[[fldf:.*]]
  ! CHECK: fir.load %[[coor6]] : !fir.ref<f32>
  print *, x(1,2,3,4,5)%a%b%c%d(6)%e(7)%f
end subroutine

! -----------------------------------------------------------------------------
!            Test a recursive derived type reference
! -----------------------------------------------------------------------------

! CHECK: func @_QMpcompPtest_recursive
! CHECK-SAME: (%[[x:.*]]: {{.*}})
subroutine test_recursive(x)
  type t
    integer :: i
    type(t), pointer :: next
  end type
  type(t) :: x

  ! CHECK: %[[fldNext1:.*]] = fir.field_index next
  ! CHECK: %[[next1:.*]] = fir.coordinate_of %[[x]], %[[fldNext1]]
  ! CHECK: %[[nextBox1:.*]] = fir.load %[[next1]]
  ! CHECK: %[[fldNext2:.*]] = fir.field_index next
  ! CHECK: %[[next2:.*]] = fir.coordinate_of %[[nextBox1]], %[[fldNext2]]
  ! CHECK: %[[nextBox2:.*]] = fir.load %[[next2]]
  ! CHECK: %[[fldNext3:.*]] = fir.field_index next
  ! CHECK: %[[next3:.*]] = fir.coordinate_of %[[nextBox2]], %[[fldNext3]]
  ! CHECK: %[[nextBox3:.*]] = fir.load %[[next3]]
  ! CHECK: %[[fldi:.*]] = fir.field_index i
  ! CHECK: %[[i:.*]] = fir.coordinate_of %[[nextBox3]], %[[fldi]]
  ! CHECK: %[[nextBox3:.*]] = fir.load %[[i]] : !fir.ref<i32>
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
    ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEreal_target)
    ! CHECK: %[[box:.*]] = fir.embox %[[target]] : (!fir.ref<f32>) -> !fir.box<f32>
    ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTreal_p0{p:!fir.box<!fir.ptr<f32>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(real_p0) :: arp0 = real_p0(real_target)

! CHECK-LABEL: fir.global @_QMpinitEbrp1 : !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}> {
! CHECK:         %[[VAL_0:.*]] = fir.undefined !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_1:.*]] = fir.field_index p, !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QMpcompEreal_array_target) : !fir.ref<!fir.array<100xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_7]] : index
! CHECK:         %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_9]] : index
! CHECK:         %[[VAL_15:.*]] = arith.divsi %[[VAL_14]], %[[VAL_9]] : index
! CHECK:         %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_12]] : index
! CHECK:         %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_12]] : index
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_19:.*]] = fir.slice %[[VAL_7]], %[[VAL_11]], %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_20:.*]] = fir.embox %[[VAL_2]](%[[VAL_18]]) {{\[}}%[[VAL_19]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<9xf32>>
! CHECK:         %[[VAL_21:.*]] = fir.rebox %[[VAL_20]] : (!fir.box<!fir.array<9xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         %[[VAL_22:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_21]], ["p", !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>] : (!fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:         fir.has_value %[[VAL_22]] : !fir.type<_QMpcompTreal_p1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:       }
  type(real_p1) :: brp1 = real_p1(real_array_target(10:50:5))

  ! CHECK-LABEL: fir.global {{.*}}@_QMpinitEccp0
    ! CHECK-DAG: %[[undef:.*]] = fir.undefined
    ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEchar_target)
    ! CHECK: %[[box:.*]] = fir.embox %[[target]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
    ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTcst_char_p0{p:!fir.box<!fir.ptr<!fir.char<1,10>>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(cst_char_p0) :: ccp0 = cst_char_p0(char_target)

  ! CHECK-LABEL: fir.global {{.*}}@_QMpinitEdcp1
    ! CHECK-DAG: %[[undef:.*]] = fir.undefined
    ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
    ! CHECK-DAG: %[[target:.*]] = fir.address_of(@_QMpcompEchar_array_target)
    ! CHECK-DAG: %[[shape:.*]] = fir.shape %c100{{.*}}
    ! CHECK-DAG: %[[box:.*]] = fir.embox %[[target]](%[[shape]]) : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<100x!fir.char<1,10>>>
    ! CHECK-DAG: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<100x!fir.char<1,10>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
    ! CHECK: %[[insert:.*]] = fir.insert_value %[[undef]], %[[rebox]], ["p", !fir.type<_QMpcompTdef_char_p1{p:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>] :
    ! CHECK: fir.has_value %[[insert]]
  type(def_char_p1) :: dcp1 = def_char_p1(char_array_target)
end module
