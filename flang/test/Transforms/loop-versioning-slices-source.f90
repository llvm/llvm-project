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
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -fdefault-integer-8 -fdefault-real-8 \
! RUN:   -frepack-arrays -frepack-arrays-contiguity=whole \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=WIDE-REPACK --enable-var-scope

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
! DRIVER: fir.do_loop
! DRIVER: %[[GRAPH_D0:.*]]:3 = fir.box_dims %[[GRAPH_DESC:[^,]+]],
! DRIVER: %[[GRAPH_SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[GRAPH_PRED:.*]] = arith.cmpi eq, %[[GRAPH_D0]]#2, %[[GRAPH_SIZE]] : index
! DRIVER: fir.if %[[GRAPH_PRED]]
! DRIVER: %[[GRAPH_BOX:.*]] = fir.convert %[[GRAPH_DESC]]
! DRIVER: %[[GRAPH_BYTES:.*]] = fir.box_addr %[[GRAPH_BOX]]
! DRIVER-SAME: -> !fir.ref<!fir.array<?xi8>>
! DRIVER: %[[GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[GRAPH_BYTES]],
! DRIVER-NEXT: %[[GRAPH_FAST:.*]] = fir.convert %[[GRAPH_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GRAPH_FAST]])
! DRIVER: } else {
! DRIVER: %[[GRAPH_FALLBACK:.*]] = fir.array_coor %[[GRAPH_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GRAPH_FALLBACK]])
! DRIVER: fir.do_loop
! DRIVER: %[[GABOR_D0:.*]]:3 = fir.box_dims %[[GABOR_DESC:[^,]+]],
! DRIVER: %[[GABOR_SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[GABOR_PRED:.*]] = arith.cmpi eq, %[[GABOR_D0]]#2, %[[GABOR_SIZE]] : index
! DRIVER: fir.if %[[GABOR_PRED]]
! DRIVER: %[[GABOR_BOX:.*]] = fir.convert %[[GABOR_DESC]]
! DRIVER: %[[GABOR_BYTES:.*]] = fir.box_addr %[[GABOR_BOX]]
! DRIVER: %[[GABOR_ADDRESS:.*]] = fir.coordinate_of %[[GABOR_BYTES]],
! DRIVER-NEXT: %[[GABOR_FAST:.*]] = fir.convert %[[GABOR_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GABOR_FAST]])
! DRIVER: } else {
! DRIVER: %[[GABOR_FALLBACK:.*]] = fir.array_coor %[[GABOR_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GABOR_FALLBACK]])
! DRIVER-LABEL: func.func @_QPrepacked_slice(
! DRIVER: fir.do_loop
! DRIVER: %[[REPACKED_D0:.*]]:3 = fir.box_dims %[[REPACKED_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[REPACKED_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[REPACKED_BOX:.*]] = fir.convert %[[REPACKED_DESC]]
! DRIVER: %[[REPACKED_BYTES:.*]] = fir.box_addr %[[REPACKED_BOX]]
! DRIVER: %[[REPACKED_ADDRESS:.*]] = fir.coordinate_of %[[REPACKED_BYTES]],
! DRIVER-NEXT: %[[REPACKED_FAST:.*]] = fir.convert %[[REPACKED_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[REPACKED_FAST]])
! DRIVER: } else {
! DRIVER: %[[REPACKED_FALLBACK:.*]] = fir.array_coor %[[REPACKED_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[REPACKED_FALLBACK]])
! DRIVER-LABEL: func.func @_QPoffset_slices(
! DRIVER: fir.do_loop
! DRIVER: %[[OFFSET_GRAPH_D0:.*]]:3 = fir.box_dims %[[OFFSET_GRAPH_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[OFFSET_GRAPH_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[OFFSET_GRAPH_BOX:.*]] = fir.convert %[[OFFSET_GRAPH_DESC]]
! DRIVER: %[[OFFSET_GRAPH_BYTES:.*]] = fir.box_addr %[[OFFSET_GRAPH_BOX]]
! DRIVER: %[[OFFSET_GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GRAPH_BYTES]],
! DRIVER-NEXT: %[[OFFSET_GRAPH_FAST:.*]] = fir.convert %[[OFFSET_GRAPH_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GRAPH_FAST]])
! DRIVER: } else {
! DRIVER: %[[OFFSET_GRAPH_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GRAPH_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GRAPH_FALLBACK]])
! DRIVER: fir.do_loop
! DRIVER: %[[OFFSET_GABOR_D0:.*]]:3 = fir.box_dims %[[OFFSET_GABOR_DESC:[^,]+]],
! DRIVER: %[[OFFSET_GABOR_SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[OFFSET_GABOR_PRED:.*]] = arith.cmpi eq, %[[OFFSET_GABOR_D0]]#2, %[[OFFSET_GABOR_SIZE]] : index
! DRIVER: fir.if %[[OFFSET_GABOR_PRED]]
! DRIVER: %[[OFFSET_GABOR_BOX:.*]] = fir.convert %[[OFFSET_GABOR_DESC]]
! DRIVER: %[[OFFSET_GABOR_BYTES:.*]] = fir.box_addr %[[OFFSET_GABOR_BOX]]
! DRIVER: %[[OFFSET_GABOR_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GABOR_BYTES]],
! DRIVER-NEXT: %[[OFFSET_GABOR_FAST:.*]] = fir.convert %[[OFFSET_GABOR_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GABOR_FAST]])
! DRIVER: } else {
! DRIVER: %[[OFFSET_GABOR_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GABOR_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[OFFSET_GABOR_FALLBACK]])
! DRIVER-LABEL: func.func @_QPrank2_patterns(
! DRIVER: fir.do_loop
! DRIVER: %[[RANK2_D0:.*]]:3 = fir.box_dims %[[RANK2_DESC:[^,]+]],
! DRIVER: %[[RANK2_SIZE0:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[RANK2_PRED0:.*]] = arith.cmpi eq, %[[RANK2_D0]]#2, %[[RANK2_SIZE0]] : index
! DRIVER: fir.if %[[RANK2_PRED0]]
! DRIVER: %[[RANK2_BOX0:.*]] = fir.convert %[[RANK2_DESC]]
! DRIVER: %[[RANK2_BYTES0:.*]] = fir.box_addr %[[RANK2_BOX0]]
! DRIVER: %[[RANK2_ADDRESS0:.*]] = fir.coordinate_of %[[RANK2_BYTES0]],
! DRIVER-NEXT: %[[RANK2_FAST0:.*]] = fir.convert %[[RANK2_ADDRESS0]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FAST0]])
! DRIVER: } else {
! DRIVER: %[[RANK2_FALLBACK0:.*]] = fir.array_coor %[[RANK2_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FALLBACK0]])
! DRIVER: fir.do_loop
! DRIVER: %[[RANK2_D1:.*]]:3 = fir.box_dims %[[RANK2_DESC]],
! DRIVER: %[[RANK2_SIZE1:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[RANK2_PRED1:.*]] = arith.cmpi eq, %[[RANK2_D1]]#2, %[[RANK2_SIZE1]] : index
! DRIVER: fir.if %[[RANK2_PRED1]]
! DRIVER: %[[RANK2_BOX1:.*]] = fir.convert %[[RANK2_DESC]]
! DRIVER: %[[RANK2_BYTES1:.*]] = fir.box_addr %[[RANK2_BOX1]]
! DRIVER: %[[RANK2_ADDRESS1:.*]] = fir.coordinate_of %[[RANK2_BYTES1]],
! DRIVER-NEXT: %[[RANK2_FAST1:.*]] = fir.convert %[[RANK2_ADDRESS1]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FAST1]])
! DRIVER: } else {
! DRIVER: %[[RANK2_FALLBACK1:.*]] = fir.array_coor %[[RANK2_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[RANK2_FALLBACK1]])
! DRIVER-LABEL: func.func @_QPgeneralized_slice(
! DRIVER: fir.do_loop
! DRIVER: %[[GENERAL_D0:.*]]:3 = fir.box_dims %[[GENERAL_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 8 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[GENERAL_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[GENERAL_BOX:.*]] = fir.convert %[[GENERAL_DESC]]
! DRIVER: %[[GENERAL_BYTES:.*]] = fir.box_addr %[[GENERAL_BOX]]
! DRIVER: %[[GENERAL_ADDRESS:.*]] = fir.coordinate_of %[[GENERAL_BYTES]],
! DRIVER-NEXT: %[[GENERAL_FAST:.*]] = fir.convert %[[GENERAL_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FAST]])
! DRIVER: } else {
! DRIVER: %[[GENERAL_FALLBACK:.*]] = fir.array_coor %[[GENERAL_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FALLBACK]])
! DRIVER-LABEL: func.func @_QPconstant_scalar(
! DRIVER: fir.do_loop
! DRIVER: %[[CONSTANT_D0:.*]]:3 = fir.box_dims %[[CONSTANT_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[CONSTANT_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[CONSTANT_BOX:.*]] = fir.convert %[[CONSTANT_DESC]]
! DRIVER: %[[CONSTANT_BYTES:.*]] = fir.box_addr %[[CONSTANT_BOX]]
! DRIVER: %[[CONSTANT_ADDRESS:.*]] = fir.coordinate_of %[[CONSTANT_BYTES]],
! DRIVER-NEXT: %[[CONSTANT_FAST:.*]] = fir.convert %[[CONSTANT_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[CONSTANT_FAST]])
! DRIVER: } else {
! DRIVER: %[[CONSTANT_FALLBACK:.*]] = fir.array_coor %[[CONSTANT_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[CONSTANT_FALLBACK]])
! DRIVER-LABEL: func.func @_QPisolated_descriptors(
! DRIVER: fir.do_loop
! DRIVER: %[[GOOD_D0:.*]]:3 = fir.box_dims %[[GOOD_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[GOOD_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[GOOD_BOX:.*]] = fir.convert %[[GOOD_DESC]]
! DRIVER: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BOX]]
! DRIVER: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
! DRIVER-NEXT: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GOOD_FAST]])
! DRIVER: } else {
! DRIVER: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[GOOD_FALLBACK]])
! DRIVER-LABEL: func.func @_QPsequential_owners(
! DRIVER: fir.do_loop
! DRIVER: %[[SEQUENTIAL_D0:.*]]:3 = fir.box_dims %[[SEQUENTIAL_DESC:[^,]+]],
! DRIVER: %[[SIZE:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[SEQUENTIAL_D0]]#2, %[[SIZE]] : index
! DRIVER: fir.if %[[PRED]]
! DRIVER: %[[SEQUENTIAL_BOX0:.*]] = fir.convert %[[SEQUENTIAL_DESC]]
! DRIVER: %[[SEQUENTIAL_BYTES0:.*]] = fir.box_addr %[[SEQUENTIAL_BOX0]]
! DRIVER: %[[SEQUENTIAL_ADDRESS0:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES0]],
! DRIVER-NEXT: %[[SEQUENTIAL_FAST0:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS0]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FAST0]])
! DRIVER: } else {
! DRIVER: %[[SEQUENTIAL_FALLBACK0:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FALLBACK0]])
! DRIVER: fir.do_loop
! DRIVER: %[[SEQUENTIAL_D1:.*]]:3 = fir.box_dims %[[SEQUENTIAL_DESC]],
! DRIVER: %[[SEQUENTIAL_SIZE1:.*]] = arith.constant 4 : index
! DRIVER-NEXT: %[[SEQUENTIAL_PRED1:.*]] = arith.cmpi eq, %[[SEQUENTIAL_D1]]#2, %[[SEQUENTIAL_SIZE1]] : index
! DRIVER: fir.if %[[SEQUENTIAL_PRED1]]
! DRIVER: %[[SEQUENTIAL_BOX1:.*]] = fir.convert %[[SEQUENTIAL_DESC]]
! DRIVER: %[[SEQUENTIAL_BYTES1:.*]] = fir.box_addr %[[SEQUENTIAL_BOX1]]
! DRIVER: %[[SEQUENTIAL_ADDRESS1:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES1]],
! DRIVER-NEXT: %[[SEQUENTIAL_FAST1:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS1]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FAST1]])
! DRIVER: } else {
! DRIVER: %[[SEQUENTIAL_FALLBACK1:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]
! DRIVER-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[SEQUENTIAL_FALLBACK1]])

! WIDE-LABEL: func.func @_QPfacerec_slices(
! WIDE-SAME: !fir.box<!fir.array<?x?x?xf64>>
! WIDE: %[[WIDE_GABOR:.*]] = fir.declare {{.*}}uniq_name = "_QFfacerec_slicesEgabor"
! WIDE: %[[WIDE_GABOR_REBOX:.*]] = fir.rebox %[[WIDE_GABOR]]
! WIDE: %[[WIDE_GRAPH:.*]] = fir.declare {{.*}}uniq_name = "_QFfacerec_slicesEgraph"
! WIDE: %[[WIDE_GRAPH_REBOX:.*]] = fir.rebox %[[WIDE_GRAPH]]
! WIDE: fir.box_dims %[[WIDE_GRAPH_REBOX]],
! WIDE: %[[WIDE_D0:.*]]:3 = fir.box_dims %[[WIDE_GRAPH]],
! WIDE: %[[WIDE_SIZE:.*]] = arith.constant 8 : index
! WIDE: %[[WIDE_PRED:.*]] = arith.cmpi eq, %[[WIDE_D0]]#2, %[[WIDE_SIZE]] : index
! WIDE: fir.if %[[WIDE_PRED]]
! WIDE: %[[WIDE_GRAPH_BOX:.*]] = fir.convert %[[WIDE_GRAPH]]
! WIDE: %[[WIDE_GRAPH_BYTES:.*]] = fir.box_addr %[[WIDE_GRAPH_BOX]]
! WIDE: %[[WIDE_ADDRESS:.*]] = fir.coordinate_of %[[WIDE_GRAPH_BYTES]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! WIDE-NEXT: %[[WIDE_FAST:.*]] = fir.convert %[[WIDE_ADDRESS]] : (!fir.ref<i8>) -> !fir.ref<f64>
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_FAST]])
! WIDE: } else {
! WIDE: %[[WIDE_FALLBACK:.*]] = fir.array_coor %[[WIDE_GRAPH]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_FALLBACK]])
! WIDE: fir.do_loop
! WIDE: %[[WIDE_GABOR_D0:.*]]:3 = fir.box_dims %[[WIDE_GABOR]],
! WIDE: %[[WIDE_GABOR_SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[WIDE_GABOR_PRED:.*]] = arith.cmpi eq, %[[WIDE_GABOR_D0]]#2, %[[WIDE_GABOR_SIZE]] : index
! WIDE: fir.if %[[WIDE_GABOR_PRED]]
! WIDE: %[[WIDE_GABOR_BOX:.*]] = fir.convert %[[WIDE_GABOR]]
! WIDE: %[[WIDE_GABOR_BYTES:.*]] = fir.box_addr %[[WIDE_GABOR_BOX]]
! WIDE: %[[WIDE_GABOR_ADDRESS:.*]] = fir.coordinate_of %[[WIDE_GABOR_BYTES]],
! WIDE-NEXT: %[[WIDE_GABOR_FAST:.*]] = fir.convert %[[WIDE_GABOR_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_GABOR_FAST]])
! WIDE: } else {
! WIDE: %[[WIDE_GABOR_FALLBACK:.*]] = fir.array_coor %[[WIDE_GABOR]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[WIDE_GABOR_FALLBACK]])
! WIDE-LABEL: func.func @_QPrepacked_slice(
! WIDE: fir.do_loop
! WIDE: %[[REPACKED_D0:.*]]:3 = fir.box_dims %[[REPACKED_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[REPACKED_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[REPACKED_BOX:.*]] = fir.convert %[[REPACKED_DESC]]
! WIDE: %[[REPACKED_BYTES:.*]] = fir.box_addr %[[REPACKED_BOX]]
! WIDE: %[[REPACKED_ADDRESS:.*]] = fir.coordinate_of %[[REPACKED_BYTES]],
! WIDE-NEXT: %[[REPACKED_FAST:.*]] = fir.convert %[[REPACKED_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[REPACKED_FAST]])
! WIDE: } else {
! WIDE: %[[REPACKED_FALLBACK:.*]] = fir.array_coor %[[REPACKED_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[REPACKED_FALLBACK]])
! WIDE-LABEL: func.func @_QPoffset_slices(
! WIDE: fir.do_loop
! WIDE: %[[OFFSET_GRAPH_D0:.*]]:3 = fir.box_dims %[[OFFSET_GRAPH_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[OFFSET_GRAPH_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[OFFSET_GRAPH_BOX:.*]] = fir.convert %[[OFFSET_GRAPH_DESC]]
! WIDE: %[[OFFSET_GRAPH_BYTES:.*]] = fir.box_addr %[[OFFSET_GRAPH_BOX]]
! WIDE: %[[OFFSET_GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GRAPH_BYTES]],
! WIDE-NEXT: %[[OFFSET_GRAPH_FAST:.*]] = fir.convert %[[OFFSET_GRAPH_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[OFFSET_GRAPH_FAST]])
! WIDE: } else {
! WIDE: %[[OFFSET_GRAPH_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GRAPH_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[OFFSET_GRAPH_FALLBACK]])
! WIDE: fir.do_loop
! WIDE: %[[OFFSET_GABOR_D0:.*]]:3 = fir.box_dims %[[OFFSET_GABOR_DESC:[^,]+]],
! WIDE: %[[OFFSET_GABOR_SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[OFFSET_GABOR_PRED:.*]] = arith.cmpi eq, %[[OFFSET_GABOR_D0]]#2, %[[OFFSET_GABOR_SIZE]] : index
! WIDE: fir.if %[[OFFSET_GABOR_PRED]]
! WIDE: %[[OFFSET_GABOR_BOX:.*]] = fir.convert %[[OFFSET_GABOR_DESC]]
! WIDE: %[[OFFSET_GABOR_BYTES:.*]] = fir.box_addr %[[OFFSET_GABOR_BOX]]
! WIDE: %[[OFFSET_GABOR_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GABOR_BYTES]],
! WIDE-NEXT: %[[OFFSET_GABOR_FAST:.*]] = fir.convert %[[OFFSET_GABOR_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[OFFSET_GABOR_FAST]])
! WIDE: } else {
! WIDE: %[[OFFSET_GABOR_FALLBACK:.*]] = fir.array_coor %[[OFFSET_GABOR_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[OFFSET_GABOR_FALLBACK]])
! WIDE-LABEL: func.func @_QPrank2_patterns(
! WIDE: fir.do_loop
! WIDE: %[[RANK2_D0:.*]]:3 = fir.box_dims %[[RANK2_DESC:[^,]+]],
! WIDE: %[[RANK2_SIZE0:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[RANK2_PRED0:.*]] = arith.cmpi eq, %[[RANK2_D0]]#2, %[[RANK2_SIZE0]] : index
! WIDE: fir.if %[[RANK2_PRED0]]
! WIDE: %[[RANK2_BOX0:.*]] = fir.convert %[[RANK2_DESC]]
! WIDE: %[[RANK2_BYTES0:.*]] = fir.box_addr %[[RANK2_BOX0]]
! WIDE: %[[RANK2_ADDRESS0:.*]] = fir.coordinate_of %[[RANK2_BYTES0]],
! WIDE-NEXT: %[[RANK2_FAST0:.*]] = fir.convert %[[RANK2_ADDRESS0]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[RANK2_FAST0]])
! WIDE: } else {
! WIDE: %[[RANK2_FALLBACK0:.*]] = fir.array_coor %[[RANK2_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[RANK2_FALLBACK0]])
! WIDE: fir.do_loop
! WIDE: %[[RANK2_D1:.*]]:3 = fir.box_dims %[[RANK2_DESC]],
! WIDE: %[[RANK2_SIZE1:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[RANK2_PRED1:.*]] = arith.cmpi eq, %[[RANK2_D1]]#2, %[[RANK2_SIZE1]] : index
! WIDE: fir.if %[[RANK2_PRED1]]
! WIDE: %[[RANK2_BOX1:.*]] = fir.convert %[[RANK2_DESC]]
! WIDE: %[[RANK2_BYTES1:.*]] = fir.box_addr %[[RANK2_BOX1]]
! WIDE: %[[RANK2_ADDRESS1:.*]] = fir.coordinate_of %[[RANK2_BYTES1]],
! WIDE-NEXT: %[[RANK2_FAST1:.*]] = fir.convert %[[RANK2_ADDRESS1]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[RANK2_FAST1]])
! WIDE: } else {
! WIDE: %[[RANK2_FALLBACK1:.*]] = fir.array_coor %[[RANK2_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[RANK2_FALLBACK1]])
! WIDE-LABEL: func.func @_QPgeneralized_slice(
! WIDE: fir.do_loop
! WIDE: %[[GENERAL_D0:.*]]:3 = fir.box_dims %[[GENERAL_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[GENERAL_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[GENERAL_BOX:.*]] = fir.convert %[[GENERAL_DESC]]
! WIDE: %[[GENERAL_BYTES:.*]] = fir.box_addr %[[GENERAL_BOX]]
! WIDE: %[[GENERAL_ADDRESS:.*]] = fir.coordinate_of %[[GENERAL_BYTES]],
! WIDE-NEXT: %[[GENERAL_FAST:.*]] = fir.convert %[[GENERAL_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FAST]])
! WIDE: } else {
! WIDE: %[[GENERAL_FALLBACK:.*]] = fir.array_coor %[[GENERAL_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FALLBACK]])
! WIDE-LABEL: func.func @_QPconstant_scalar(
! WIDE: fir.do_loop
! WIDE: %[[CONSTANT_D0:.*]]:3 = fir.box_dims %[[CONSTANT_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[CONSTANT_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[CONSTANT_BOX:.*]] = fir.convert %[[CONSTANT_DESC]]
! WIDE: %[[CONSTANT_BYTES:.*]] = fir.box_addr %[[CONSTANT_BOX]]
! WIDE: %[[CONSTANT_ADDRESS:.*]] = fir.coordinate_of %[[CONSTANT_BYTES]],
! WIDE-NEXT: %[[CONSTANT_FAST:.*]] = fir.convert %[[CONSTANT_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[CONSTANT_FAST]])
! WIDE: } else {
! WIDE: %[[CONSTANT_FALLBACK:.*]] = fir.array_coor %[[CONSTANT_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[CONSTANT_FALLBACK]])
! WIDE-LABEL: func.func @_QPisolated_descriptors(
! WIDE: fir.do_loop
! WIDE: %[[GOOD_D0:.*]]:3 = fir.box_dims %[[GOOD_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[GOOD_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[GOOD_BOX:.*]] = fir.convert %[[GOOD_DESC]]
! WIDE: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BOX]]
! WIDE: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
! WIDE-NEXT: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GOOD_FAST]])
! WIDE: } else {
! WIDE: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GOOD_FALLBACK]])
! WIDE-LABEL: func.func @_QPsequential_owners(
! WIDE: fir.do_loop
! WIDE: %[[SEQUENTIAL_D0:.*]]:3 = fir.box_dims %[[SEQUENTIAL_DESC:[^,]+]],
! WIDE: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[PRED:.*]] = arith.cmpi eq, %[[SEQUENTIAL_D0]]#2, %[[SIZE]] : index
! WIDE: fir.if %[[PRED]]
! WIDE: %[[SEQUENTIAL_BOX0:.*]] = fir.convert %[[SEQUENTIAL_DESC]]
! WIDE: %[[SEQUENTIAL_BYTES0:.*]] = fir.box_addr %[[SEQUENTIAL_BOX0]]
! WIDE: %[[SEQUENTIAL_ADDRESS0:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES0]],
! WIDE-NEXT: %[[SEQUENTIAL_FAST0:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS0]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[SEQUENTIAL_FAST0]])
! WIDE: } else {
! WIDE: %[[SEQUENTIAL_FALLBACK0:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[SEQUENTIAL_FALLBACK0]])
! WIDE: fir.do_loop
! WIDE: %[[SEQUENTIAL_D1:.*]]:3 = fir.box_dims %[[SEQUENTIAL_DESC]],
! WIDE: %[[SEQUENTIAL_SIZE1:.*]] = arith.constant 8 : index
! WIDE-NEXT: %[[SEQUENTIAL_PRED1:.*]] = arith.cmpi eq, %[[SEQUENTIAL_D1]]#2, %[[SEQUENTIAL_SIZE1]] : index
! WIDE: fir.if %[[SEQUENTIAL_PRED1]]
! WIDE: %[[SEQUENTIAL_BOX1:.*]] = fir.convert %[[SEQUENTIAL_DESC]]
! WIDE: %[[SEQUENTIAL_BYTES1:.*]] = fir.box_addr %[[SEQUENTIAL_BOX1]]
! WIDE: %[[SEQUENTIAL_ADDRESS1:.*]] = fir.coordinate_of %[[SEQUENTIAL_BYTES1]],
! WIDE-NEXT: %[[SEQUENTIAL_FAST1:.*]] = fir.convert %[[SEQUENTIAL_ADDRESS1]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[SEQUENTIAL_FAST1]])
! WIDE: } else {
! WIDE: %[[SEQUENTIAL_FALLBACK1:.*]] = fir.array_coor %[[SEQUENTIAL_DESC]]
! WIDE-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[SEQUENTIAL_FALLBACK1]])

! REPACK-LABEL: func.func @_QPrepacked_slice(
! REPACK: %[[PACKED:.*]] = fir.pack_array %[[ORIGINAL:.*]] heap whole
! REPACK: %[[DECLARED:.*]] = fir.declare %[[PACKED]]
! REPACK: fir.do_loop
! REPACK: %[[SIZE:.*]] = arith.constant 4 : index
! REPACK-NEXT: %[[PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %[[SIZE]] : index
! REPACK: fir.if %[[PRED]]
! REPACK: %[[BYTE_BOX:.*]] = fir.convert %[[DECLARED]]
! REPACK-SAME: -> !fir.box<!fir.array<?xi8>>
! REPACK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
! REPACK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! REPACK-NEXT: %[[FAST:.*]] = fir.convert %[[BYTE_ADDRESS]] : (!fir.ref<i8>) -> !fir.ref<f32>
! REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[FAST]])
! REPACK: } else {
! REPACK: %[[FALLBACK:.*]] = fir.array_coor %[[DECLARED]]
! REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[FALLBACK]])
! REPACK: fir.unpack_array %[[PACKED]] to %[[ORIGINAL]] heap
! REPACK-LABEL: func.func @_QPoffset_slices(

! WIDE-REPACK-LABEL: func.func @_QPrepacked_slice(
! WIDE-REPACK-SAME: !fir.box<!fir.array<?x?xf64>>
! WIDE-REPACK: %[[PACKED:.*]] = fir.pack_array %[[ORIGINAL:.*]] heap whole
! WIDE-REPACK: %[[DECLARED:.*]] = fir.declare %[[PACKED]]
! WIDE-REPACK: fir.do_loop
! WIDE-REPACK: %[[SIZE:.*]] = arith.constant 8 : index
! WIDE-REPACK-NEXT: %[[PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %[[SIZE]] : index
! WIDE-REPACK: fir.if %[[PRED]]
! WIDE-REPACK: %[[BYTE_BOX:.*]] = fir.convert %[[DECLARED]]
! WIDE-REPACK-SAME: -> !fir.box<!fir.array<?xi8>>
! WIDE-REPACK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
! WIDE-REPACK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! WIDE-REPACK-NEXT: %[[FAST:.*]] = fir.convert %[[BYTE_ADDRESS]] : (!fir.ref<i8>) -> !fir.ref<f64>
! WIDE-REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[FAST]])
! WIDE-REPACK: } else {
! WIDE-REPACK: %[[FALLBACK:.*]] = fir.array_coor %[[DECLARED]]
! WIDE-REPACK-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[FALLBACK]])
! WIDE-REPACK: fir.unpack_array %[[PACKED]] to %[[ORIGINAL]] heap
! WIDE-REPACK-LABEL: func.func @_QPoffset_slices(

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
