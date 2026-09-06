! RUN: %flang_fc1 -emit-fir -O3 %s -o - | \
! RUN:   fir-opt --verify-each --loop-versioning | \
! RUN:   FileCheck %s --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=DRIVER --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -fdefault-integer-8 -fdefault-real-8 \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=WIDE --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -frepack-arrays -frepack-arrays-contiguity=whole \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=REPACK --enable-var-scope

! Verify that source-expressible slice forms reach the byte-address fast path
! through both the fc1 and driver pipelines. The additional driver modes make
! element width and frontend-generated repacking observable.

! Source-level rank-3 and rank-4 forms from the facerec expression. I/O keeps
! the slices attached to fir.array_coor operations, so this test connects
! frontend lowering to both the new byte fast path and the sliced fallback.
subroutine facerec_slices(graph, gabor, indices, y)
  implicit none
  real, intent(inout) :: graph(:, :, :), gabor(:, :, :, :)
  integer, intent(in) :: indices(2)
  integer(kind=8), intent(in) :: y

  read(*, *) graph(:, :, indices), gabor(:, :, indices, y)
end subroutine

! Keep the frontend wrapper configuration in a compiler test. The runtime test
! separately verifies that values written through the repacked fast path are
! copied back to a noncontiguous actual argument.
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
! receive independent address plans.
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

! An unsupported dynamic-step descriptor must remain generic without blocking
! an independent descriptor whose section step is statically one.
subroutine isolated_descriptors(good, strided, indices, step)
  implicit none
  real, intent(inout) :: good(:, :), strided(:, :)
  integer, intent(in) :: indices(2), step

  read(*, *) good(2:4, indices)
  read(*, *) strided(1:5:step, indices)
end subroutine

! Sequential owners of one descriptor require separate frozen address facts so
! each generated fast loop uses its own slice.
subroutine sequential_owners(values, indices)
  implicit none
  real, intent(inout) :: values(:, :)
  integer, intent(in) :: indices(2)

  read(*, *) values(1:2, indices)
  read(*, *) values(4:5, indices)
end subroutine

! CHECK-LABEL: func.func @_QPfacerec_slices(
! CHECK: %[[GRAPH_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[GRAPH_BOX:.*]] = fir.convert %[[GRAPH_DESC:[^ ]+]]
! CHECK: %[[GRAPH_BYTES:.*]] = fir.box_addr %[[GRAPH_BOX]]
! CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
! CHECK: %[[GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[GRAPH_BYTES]],
! CHECK: %[[GRAPH_FAST:.*]] = fir.convert %[[GRAPH_ADDRESS]]
! CHECK-SAME: -> !fir.ref<f32>
! CHECK: fir.call @_FortranAioInputReal32({{.*}}, %[[GRAPH_FAST]])
! CHECK: } else {
! CHECK: %[[GRAPH_FALLBACK:.*]] = fir.array_coor %[[GRAPH_DESC]]{{.*}}[%[[GRAPH_SLICE]]]
! CHECK: fir.call @_FortranAioInputReal32({{.*}}, %[[GRAPH_FALLBACK]])
! CHECK: %[[GABOR_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[GABOR_BOX:.*]] = fir.convert %[[GABOR_DESC:[^ ]+]]
! CHECK: %[[GABOR_BYTES:.*]] = fir.box_addr %[[GABOR_BOX]]
! CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
! CHECK: %[[GABOR_ADDRESS:.*]] = fir.coordinate_of %[[GABOR_BYTES]],
! CHECK: %[[GABOR_FAST:.*]] = fir.convert %[[GABOR_ADDRESS]]
! CHECK-SAME: -> !fir.ref<f32>
! CHECK: fir.call @_FortranAioInputReal32({{.*}}, %[[GABOR_FAST]])
! CHECK: } else {
! CHECK: %[[GABOR_FALLBACK:.*]] = fir.array_coor %[[GABOR_DESC]]{{.*}}[%[[GABOR_SLICE]]]
! CHECK: fir.call @_FortranAioInputReal32({{.*}}, %[[GABOR_FALLBACK]])

! DRIVER: IR Dump After LoopVersioning
! DRIVER-LABEL: func.func @_QPfacerec_slices(
! DRIVER: fir.if
! DRIVER: !fir.ref<!fir.array<?xi8>>
! DRIVER: fir.coordinate_of
! DRIVER: } else {
! DRIVER: fir.array_coor
! DRIVER-LABEL: func.func @_QPrepacked_slice(

! WIDE-LABEL: func.func @_QPfacerec_slices(
! WIDE-SAME: !fir.box<!fir.array<?x?x?xf64>>
! WIDE: %[[WIDE_GRAPH:.*]] = fir.declare {{.*}}uniq_name = "_QFfacerec_slicesEgraph"
! WIDE: %[[WIDE_GRAPH_REBOX:.*]] = fir.rebox %[[WIDE_GRAPH]]
! WIDE: fir.box_dims %[[WIDE_GRAPH_REBOX]],
! WIDE: %[[WIDE_D0:.*]]:3 = fir.box_dims %[[WIDE_GRAPH]],
! WIDE: %[[WIDE_SIZE:.*]] = arith.constant 8 : index
! WIDE: %[[WIDE_PRED:.*]] = arith.cmpi eq, %[[WIDE_D0]]#2, %[[WIDE_SIZE]] : index
! WIDE: fir.if %[[WIDE_PRED]]
! WIDE: %[[WIDE_ADDRESS:.*]] = fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! WIDE-NEXT: %[[WIDE_FAST:.*]] = fir.convert %[[WIDE_ADDRESS]] : (!fir.ref<i8>) -> !fir.ref<f64>
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_FAST]])
! WIDE: } else {
! WIDE: %[[WIDE_FALLBACK:.*]] = fir.array_coor
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_FALLBACK]])
! WIDE-LABEL: func.func @_QPrepacked_slice(

! REPACK-LABEL: func.func @_QPrepacked_slice(
! REPACK: %[[PACKED:.*]] = fir.pack_array
! REPACK: %[[DECLARED:.*]] = fir.declare %[[PACKED]]
! REPACK: fir.if
! REPACK: %[[BYTE_BOX:.*]] = fir.convert %[[DECLARED]]
! REPACK-SAME: -> !fir.box<!fir.array<?xi8>>
! REPACK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
! REPACK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! REPACK-NEXT: %[[FAST:.*]] = fir.convert %[[BYTE_ADDRESS]] : (!fir.ref<i8>) -> !fir.ref<f32>
! REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[FAST]])
! REPACK: } else {
! REPACK: %[[FALLBACK:.*]] = fir.array_coor %[[DECLARED]]
! REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[FALLBACK]])
! REPACK-LABEL: func.func @_QPoffset_slices(

! CHECK-LABEL: func.func @_QPoffset_slices(
! CHECK: %[[OFFSET_GRAPH_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[OFFSET_GRAPH_BOX:.*]] = fir.convert %[[OFFSET_GRAPH_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[OFFSET_GRAPH_BYTES:.*]] = fir.box_addr %[[OFFSET_GRAPH_BOX]]
! CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
! CHECK: %[[OFFSET_GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GRAPH_BYTES]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[OFFSET_GRAPH_FAST:.*]] = fir.convert %[[OFFSET_GRAPH_ADDRESS]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GRAPH_FAST]])
! CHECK: } else {
! CHECK: %[[OFFSET_GRAPH_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GRAPH_DESC]]{{.*}}[%[[OFFSET_GRAPH_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GRAPH_FALLBACK]])
! CHECK: %[[OFFSET_GABOR_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[OFFSET_GABOR_BOX:.*]] = fir.convert %[[OFFSET_GABOR_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[OFFSET_GABOR_BYTES:.*]] = fir.box_addr %[[OFFSET_GABOR_BOX]]
! CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
! CHECK: %[[OFFSET_GABOR_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GABOR_BYTES]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[OFFSET_GABOR_FAST:.*]] = fir.convert %[[OFFSET_GABOR_ADDRESS]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GABOR_FAST]])
! CHECK: } else {
! CHECK: %[[OFFSET_GABOR_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GABOR_DESC]]{{.*}}[%[[OFFSET_GABOR_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GABOR_FALLBACK]])

! CHECK-LABEL: func.func @_QPrank2_patterns(
! CHECK: %[[RANK2_SLICE0:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[RANK2_BOX0:.*]] = fir.convert %[[RANK2_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[RANK2_BYTES0:.*]] = fir.box_addr %[[RANK2_BOX0]]
! CHECK: %[[RANK2_ADDRESS0:.*]] = fir.coordinate_of %[[RANK2_BYTES0]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[RANK2_FAST0:.*]] = fir.convert %[[RANK2_ADDRESS0]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FAST0]])
! CHECK: } else {
! CHECK: %[[RANK2_FALLBACK0:.*]] = fir.array_coor %[[RANK2_DESC]]{{.*}}[%[[RANK2_SLICE0]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FALLBACK0]])
! CHECK: %[[RANK2_SLICE1:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[RANK2_BOX1:.*]] = fir.convert %[[RANK2_DESC]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[RANK2_BYTES1:.*]] = fir.box_addr %[[RANK2_BOX1]]
! CHECK: %[[RANK2_ADDRESS1:.*]] = fir.coordinate_of %[[RANK2_BYTES1]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[RANK2_FAST1:.*]] = fir.convert %[[RANK2_ADDRESS1]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FAST1]])
! CHECK: } else {
! CHECK: %[[RANK2_FALLBACK1:.*]] = fir.array_coor %[[RANK2_DESC]]{{.*}}[%[[RANK2_SLICE1]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FALLBACK1]])

! CHECK-LABEL: func.func @_QPgeneralized_slice(
! CHECK: %[[GENERAL_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[GENERAL_BOX:.*]] = fir.convert %[[GENERAL_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[GENERAL_BYTES:.*]] = fir.box_addr %[[GENERAL_BOX]]
! CHECK: %[[GENERAL_ADDRESS:.*]] = fir.coordinate_of %[[GENERAL_BYTES]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[GENERAL_FAST:.*]] = fir.convert %[[GENERAL_ADDRESS]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FAST]])
! CHECK: } else {
! CHECK: %[[GENERAL_FALLBACK:.*]] = fir.array_coor %[[GENERAL_DESC]]{{.*}}[%[[GENERAL_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FALLBACK]])

! CHECK-LABEL: func.func @_QPconstant_scalar(
! CHECK: %[[CONSTANT_SLICE:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[CONSTANT_BOX:.*]] = fir.convert %[[CONSTANT_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[CONSTANT_BYTES:.*]] = fir.box_addr %[[CONSTANT_BOX]]
! CHECK: %[[CONSTANT_ADDRESS:.*]] = fir.coordinate_of %[[CONSTANT_BYTES]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[CONSTANT_FAST:.*]] = fir.convert %[[CONSTANT_ADDRESS]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[CONSTANT_FAST]])
! CHECK: } else {
! CHECK: %[[CONSTANT_FALLBACK:.*]] = fir.array_coor %[[CONSTANT_DESC]]{{.*}}[%[[CONSTANT_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[CONSTANT_FALLBACK]])

! CHECK-LABEL: func.func @_QPisolated_descriptors(
! CHECK: %[[GOOD_SLICE:.*]] = fir.slice
! CHECK: %[[PRED:.*]] = arith.cmpi eq
! CHECK: fir.if %[[PRED]]
! CHECK: %[[GOOD_BOX:.*]] = fir.convert %[[GOOD_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BOX]]
! CHECK: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GOOD_FAST]])
! CHECK: } else {
! CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD_DESC]]{{.*}}[%[[GOOD_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GOOD_FALLBACK]])
! CHECK: %[[STRIDED_SLICE:.*]] = fir.slice
! CHECK-NOT: fir.if
! CHECK: %[[STRIDED_ACCESS:.*]] = fir.array_coor %[[STRIDED_DESC:[^ ]+]]{{.*}}[%[[STRIDED_SLICE]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[STRIDED_ACCESS]])
! CHECK-NOT: fir.if

! CHECK-LABEL: func.func @_QPsequential_owners(
! CHECK: %[[SEQUENTIAL_SLICE0:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[SEQUENTIAL_BOX0:.*]] = fir.convert %[[SEQUENTIAL_DESC:[^ ]+]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[SEQUENTIAL_BYTES0:.*]] = fir.box_addr %[[SEQUENTIAL_BOX0]]
! CHECK: %[[SEQUENTIAL_ADDRESS0:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES0]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[SEQUENTIAL_FAST0:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS0]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FAST0]])
! CHECK: } else {
! CHECK: %[[SEQUENTIAL_FALLBACK0:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]{{.*}}[%[[SEQUENTIAL_SLICE0]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FALLBACK0]])
! CHECK: %[[SEQUENTIAL_SLICE1:.*]] = fir.slice
! CHECK: fir.if
! CHECK: %[[SEQUENTIAL_BOX1:.*]] = fir.convert %[[SEQUENTIAL_DESC]]
! CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
! CHECK: %[[SEQUENTIAL_BYTES1:.*]] = fir.box_addr %[[SEQUENTIAL_BOX1]]
! CHECK: %[[SEQUENTIAL_ADDRESS1:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES1]],
! CHECK-SAME: -> !fir.ref<i8>
! CHECK-NEXT: %[[SEQUENTIAL_FAST1:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS1]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FAST1]])
! CHECK: } else {
! CHECK: %[[SEQUENTIAL_FALLBACK1:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]{{.*}}[%[[SEQUENTIAL_SLICE1]]]
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FALLBACK1]])
