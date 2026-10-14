! RUN: %flang_fc1 -emit-fir -O3 %s -o - | \
! RUN:   fir-opt --verify-each --loop-versioning | \
! RUN:   FileCheck %s --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=DRIVER --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -frepack-arrays -frepack-arrays-contiguity=whole \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=REPACK --enable-var-scope
! RUN: %flang -S -O3 -ffast-math -fstack-arrays \
! RUN:   -fversion-loops-for-stride -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null > %t.pointer 2>&1
! RUN: FileCheck %s --input-file=%t.pointer --check-prefix=POINTER \
! RUN:   --enable-var-scope
! RUN: FileCheck %s --input-file=%t.pointer --check-prefix=POINTER-GUARDS
! Verify that source-expressible slice forms reach the flattened typed fast
! path through both the fc1 and driver pipelines. The repack mode makes
! frontend-generated temporary arrays observable.

! Source-level rank-3 and rank-4 forms from the facerec expression. I/O keeps
! the slices attached to fir.array_coor operations, so this test connects
! frontend lowering to both the fast path and the sliced fallback.
subroutine facerec_slices(graph, gabor, indices, y)
  implicit none
  real, intent(inout) :: graph(:, :, :), gabor(:, :, :, :)
  integer, intent(in) :: indices(2)
  integer(kind=8), intent(in) :: y

  read(*, *) graph(:, :, indices), gabor(:, :, indices, y)
end subroutine

! The fc1 pipeline proves that every supported source form reaches the common
! typed fast path and retains the original sliced access in the fallback.
! CHECK-LABEL: func.func @_QPfacerec_slices(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPrepacked_slice(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPoffset_slices(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPrank2_patterns(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPgeneralized_slice(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPconstant_scalar(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPleading_scalar_shift(
! CHECK: fir.shift
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: } else {
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPisolated_descriptors(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK-COUNT-2: fir.array_coor {{.*}}[{{.*}}]
! CHECK-LABEL: func.func @_QPsequential_owners(
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]
! CHECK: fir.if
! CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK: fir.array_coor {{.*}}[{{.*}}]

! The driver pipeline must expose the same fast/fallback structure.
! DRIVER-LABEL: func.func @_QPfacerec_slices(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! DRIVER: fir.array_coor {{.*}}[{{.*}}]
! DRIVER: fir.if
! DRIVER: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! DRIVER: fir.array_coor {{.*}}[{{.*}}]
! DRIVER-LABEL: func.func @_QPrepacked_slice(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: } else {
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPoffset_slices(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPrank2_patterns(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPgeneralized_slice(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: } else {
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPconstant_scalar(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: } else {
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPleading_scalar_shift(
! DRIVER: fir.shift
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: } else {
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPisolated_descriptors(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER-COUNT-2: fir.array_coor
! DRIVER-LABEL: func.func @_QPsequential_owners(
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor
! DRIVER: fir.if
! DRIVER: fir.coordinate_of
! DRIVER: fir.array_coor

! The motivating pointer case needs the full driver pipeline to fold the
! frontend reboxes into sliced accesses before loop versioning.
! POINTER-LABEL: func.func @_QPpointer_stride_repro(
! POINTER: %[[DST_BOX:.*]] = fir.load
! POINTER: %[[X_BOX:.*]] = fir.load
! POINTER: %[[Y_BOX:.*]] = fir.load
! POINTER: %[[Z_BOX:.*]] = fir.load
! POINTER: fir.allocmem
! POINTER: fir.if
! POINTER-COUNT-4: fir.coordinate_of
! POINTER: } else {
! POINTER-COUNT-4: fir.array_coor

! POINTER-GUARDS-LABEL: func.func @_QPpointer_stride_repro(
! POINTER-GUARDS: fir.allocmem
! POINTER-GUARDS-DAG: %[[DIM0:[^ :]+]]:3 = fir.box_dims
! POINTER-GUARDS-DAG: %[[DIM1:[^ :]+]]:3 = fir.box_dims
! POINTER-GUARDS-DAG: %[[DIM2:[^ :]+]]:3 = fir.box_dims
! POINTER-GUARDS-DAG: %[[DIM3:[^ :]+]]:3 = fir.box_dims
! POINTER-GUARDS-DAG: arith.constant 16 : index
! POINTER-GUARDS-DAG: arith.constant 16 : index
! POINTER-GUARDS-DAG: arith.constant 16 : index
! POINTER-GUARDS-DAG: arith.constant 16 : index
! POINTER-GUARDS-DAG: arith.cmpi eq, %[[DIM0]]#2, %{{.*}} : index
! POINTER-GUARDS-DAG: arith.cmpi eq, %[[DIM1]]#2, %{{.*}} : index
! POINTER-GUARDS-DAG: arith.cmpi eq, %[[DIM2]]#2, %{{.*}} : index
! POINTER-GUARDS-DAG: arith.cmpi eq, %[[DIM3]]#2, %{{.*}} : index
! POINTER-GUARDS-NOT: arith.cmpi eq
! POINTER-GUARDS: fir.if

! Repacking must surround the same versioned fast path and copy the result
! back to the original noncontiguous argument.
! REPACK-LABEL: func.func @_QPrepacked_slice(
! REPACK: %[[PACKED:.*]] = fir.pack_array %[[ORIGINAL:.*]] heap whole
! REPACK: fir.if
! REPACK: fir.coordinate_of
! REPACK: } else {
! REPACK: fir.array_coor
! REPACK: fir.unpack_array %[[PACKED]] to %[[ORIGINAL]] heap

! Keep the frontend wrapper configuration in a compiler test. The execution
! test in llvm-test-suite/Fortran/UnitTests/loop-versioning-slices verifies that
! values written through the repacked fast path are copied back to a
! noncontiguous actual argument.
subroutine repacked_slice(values, indices)
  implicit none
  real, intent(inout) :: values(:, :)
  integer, intent(in) :: indices(2)

  read(*, *) values(2:3, indices)
end subroutine

! Non-one section lower bounds require explicit retained-section corrections;
! verify that the corresponding runtime scenario reaches the fast path.
subroutine offset_slices(graph, gabor, indices, y)
  implicit none
  real, intent(inout) :: graph(:, :, :), gabor(:, :, :, :)
  integer, intent(in) :: indices(2)
  integer(kind=8), intent(in) :: y

  read(*, *) graph(2:3, 2:3, indices), gabor(2:3, 2:3, indices, y)
end subroutine

! Two accesses to one descriptor use distinct dynamic lower bounds and must
! compute independent flattened indices.
subroutine rank2_patterns(values, lower0, lower1, indices)
  implicit none
  real, intent(inout) :: values(:, :)
  integer, intent(in) :: lower0, lower1, indices(2)

  read(*, *) values(lower0:lower0 + 1, indices), &
             values(lower1:lower1 + 1, indices)
end subroutine

! Section/Scalar/Section with eight-byte elements verifies that acceptance is
! not limited to prefix sections or the default real element size.
subroutine generalized_slice(values, lower0, lower2, indices)
  implicit none
  real(kind=8), intent(inout) :: values(:, :, :)
  integer, intent(in) :: lower0, lower2, indices(2)

  read(*, *) values(lower0:lower0 + 1, indices, lower2:lower2 + 1)
end subroutine

! A constant-one trailing scalar has zero outer contribution; the fast address
! must still select the same element as the sliced access.
subroutine constant_scalar(values, indices)
  implicit none
  real, intent(inout) :: values(:, :, :)
  integer, intent(in) :: indices(2)

  read(*, *) values(2:3, indices, 1)
end subroutine

! A leading scalar dimension and explicit dummy lower bounds exercise a
! source-generated fir.shift together with a Scalar/Section slice.
subroutine leading_scalar_shift(values, row, indices)
  implicit none
  real, intent(inout) :: values(0:, -2:)
  integer, intent(in) :: row, indices(2)

  read(*, *) values(row, indices)
end subroutine

! An unsupported dynamic-step descriptor must remain generic without blocking
! an independent descriptor whose section step is statically one.
subroutine isolated_descriptors(good, strided, indices, step)
  implicit none
  real, intent(inout) :: good(:, :), strided(:, :)
  integer, intent(in) :: indices(2), step

  read(*, *) good(2:4, indices)
  read(*, *) strided(1:5:step, indices)
end subroutine

! Sequential owners of one descriptor must each derive the flattened index
! from their own slice.
subroutine sequential_owners(values, indices)
  implicit none
  real, intent(inout) :: values(:, :)
  integer, intent(in) :: indices(2)

  read(*, *) values(1:2, indices)
  read(*, *) values(4:5, indices)
end subroutine

! Pointer descriptors are passed by address. Their loaded box must still be
! recognized as an argument, and contiguous complex(kind=8) has byte stride 16.
subroutine pointer_stride_repro(dst, x, y, z)
  implicit none
  complex(kind=8), pointer, intent(inout) :: dst(:)
  complex(kind=8), pointer, intent(in) :: x(:), y(:), z(:)

  dst(:) = dst(:) + x(:) * y(:) * z(:)
end subroutine
