! Test lowering of pointer initial target
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! This tests focus on the scope context of initial data target.
! More complete tests regarding the initial data target expression
! are done in pointer-initial-target.f90.

! Test pointer initial data target with pointer in common blocks
block data
  real, pointer :: p
  real, save, target :: b
  common /a/ p
  data p /b/
! CHECK-LABEL: fir.global @_QBa : tuple<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[undef:.*]] = fir.undefined tuple<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[b:.*]] = fir.address_of(@_QEb) : !fir.ref<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[b]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: %[[a:.*]] = fir.insert_value %[[undef]], %[[rebox]], [0 : index] : (tuple<!fir.box<!fir.ptr<f32>>>, !fir.box<!fir.ptr<f32>>) -> tuple<!fir.box<!fir.ptr<f32>>>
  ! CHECK: fir.has_value %[[a]] : tuple<!fir.box<!fir.ptr<f32>>>
end block data

! Test two common depending on each others because of initial data
! targets
block data tied
  real, target :: x1 = 42
  real, target :: x2 = 43
  real, pointer :: p1 => x2
  real, pointer :: p2 => x1
  common /c1/ x1, p1
  common /c2/ x2, p2
! CHECK-LABEL: fir.global @_QBc1 : tuple<f32, !fir.array<4xi8>, !fir.box<!fir.ptr<f32>>>
  ! CHECK: fir.address_of(@_QBc2) : !fir.ref<tuple<f32, !fir.array<4xi8>, !fir.box<!fir.ptr<f32>>>>
! CHECK-LABEL: fir.global @_QBc2 : tuple<f32, !fir.array<4xi8>, !fir.box<!fir.ptr<f32>>>
  ! CHECK: fir.address_of(@_QBc1) : !fir.ref<tuple<f32, !fir.array<4xi8>, !fir.box<!fir.ptr<f32>>>>
end block data

! Test pointer in a common with initial target in the same common.
block data bdsnake
  integer, target :: b = 42
  integer, pointer :: p => b
  common /snake/ p, b
! CHECK-LABEL: fir.global @_QBsnake : tuple<!fir.box<!fir.ptr<i32>>, i32>
  ! CHECK: %[[tuple0:.*]] = fir.undefined tuple<!fir.box<!fir.ptr<i32>>, i32>
  ! CHECK: %[[snakeAddr:.*]] = fir.address_of(@_QBsnake) : !fir.ref<tuple<!fir.box<!fir.ptr<i32>>, i32>>
  ! CHECK: %[[byteView:.*]] = fir.convert %[[snakeAddr:.*]] : (!fir.ref<tuple<!fir.box<!fir.ptr<i32>>, i32>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[byteView]], %c24{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[bAddr:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ref<i32>
  ! CHECK: %[[box:.*]] = fir.embox %[[bAddr]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<i32>) -> !fir.box<!fir.ptr<i32>>
  ! CHECK: %[[tuple1:.*]] = fir.insert_value %[[tuple0]], %[[rebox]], [0 : index] : (tuple<!fir.box<!fir.ptr<i32>>, i32>, !fir.box<!fir.ptr<i32>>) -> tuple<!fir.box<!fir.ptr<i32>>, i32>
  ! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple1]], %c42{{.*}}, [1 : index] : (tuple<!fir.box<!fir.ptr<i32>>, i32>, i32) -> tuple<!fir.box<!fir.ptr<i32>>, i32>
  ! CHECK: fir.has_value %[[tuple2]] : tuple<!fir.box<!fir.ptr<i32>>, i32>
end block data

! Test pointer initial data target in modules
module some_mod
  real, target :: x(100)
  real, pointer :: p(:) => x
! CHECK-LABEL: fir.global @_QMsome_modEp : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
  ! CHECK: %[[x:.*]] = fir.address_of(@_QMsome_modEx) : !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end module

! Test initial data target in a common block
module some_mod_2
  real, target :: x(100), y(10:209)
  common /com/ x, y
  save :: /com/
  real, pointer :: p(:) => y
! CHECK-LABEL: fir.global @_QMsome_mod_2Ep : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
  ! CHECK: %[[c:.*]] = fir.address_of(@_QBcom) : !fir.ref<!fir.array<1200xi8>>
  ! CHECK: %[[com:.*]] = fir.convert %[[c]] : (!fir.ref<!fir.array<1200xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK: %[[yRaw:.*]] = fir.coordinate_of %[[com]], %c400{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[y:.*]] = fir.convert %[[yRaw]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<200xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c10{{.*}}, %c200{{.*}} : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[y]](%[[shape]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<200xf32>>
  ! CHECK: %[[shift:.*]] = fir.shift %c10{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]](%[[shift]]) : (!fir.box<!fir.array<200xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end module
