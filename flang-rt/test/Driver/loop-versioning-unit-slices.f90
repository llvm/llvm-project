! Test FIR loop versioning by checking the generated fast path and verifying
! that its execution produces the expected results.
! Verify runtime semantics for source-expressible static-unit slices.
! The compiler source test separately proves byte fast paths for every
! distinct positive access and ownership shape exercised below.
! REQUIRES: llvm-flang
! UNSUPPORTED: offload-cuda
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=VERSIONED --enable-var-scope
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -fdefault-integer-8 -fdefault-real-8 \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=VERSIONED --enable-var-scope
! RUN: %flang %isysroot -L"%libdir" -O3 -fversion-loops-for-stride %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
! RUN: %flang %isysroot -L"%libdir" -O3 -fversion-loops-for-stride \
! RUN:   -fdefault-integer-8 -fdefault-real-8 %s -o %t.wide
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t.wide | FileCheck %s

module loop_versioning_unit_slices_m
  implicit none
contains
  subroutine fill_slices(graph, gabor, indices, y, record)
    real, intent(inout) :: graph(:, :, :), gabor(:, :, :, :)
    integer, intent(in) :: indices(2)
    integer(kind=8), intent(in) :: y
    character(*), intent(in) :: record

    read(record, *) graph(:, :, indices), gabor(:, :, indices, y)
  end subroutine

  subroutine fill_offset_slices(graph, gabor, indices, y, record)
    real, intent(inout) :: graph(:, :, :), gabor(:, :, :, :)
    integer, intent(in) :: indices(2)
    integer(kind=8), intent(in) :: y
    character(*), intent(in) :: record

    read(record, *) graph(2:3, 2:3, indices), &
                    gabor(2:3, 2:3, indices, y)
  end subroutine

  subroutine fill_rank2_patterns(values, lower0, lower1, indices, record)
    real, intent(inout) :: values(:, :)
    integer, intent(in) :: lower0, lower1, indices(2)
    character(*), intent(in) :: record

    read(record, *) values(lower0:lower0 + 1, indices), &
                    values(lower1:lower1 + 1, indices)
  end subroutine

  subroutine fill_generalized(values, lower0, lower2, indices, record)
    real(kind=8), intent(inout) :: values(:, :, :)
    integer, intent(in) :: lower0, lower2, indices(2)
    character(*), intent(in) :: record

    read(record, *) values(lower0:lower0 + 1, indices, lower2:lower2 + 1)
  end subroutine

  subroutine fill_constant_scalar(values, indices, record)
    real, intent(inout) :: values(:, :, :)
    integer, intent(in) :: indices(2)
    character(*), intent(in) :: record

    read(record, *) values(2:3, indices, 1)
  end subroutine

  subroutine fill_isolated_descriptors(good, strided, indices, step, &
                                       good_record, strided_record)
    real, intent(inout) :: good(:, :), strided(:, :)
    integer, intent(in) :: indices(2), step
    character(*), intent(in) :: good_record, strided_record

    read(good_record, *) good(2:4, indices)
    read(strided_record, *) strided(1:5:step, indices)
  end subroutine

  subroutine fill_sequential_owners(values, indices, first_record, &
                                    second_record)
    real, intent(inout) :: values(:, :)
    integer, intent(in) :: indices(2)
    character(*), intent(in) :: first_record, second_record

    read(first_record, *) values(1:2, indices)
    read(second_record, *) values(4:5, indices)
  end subroutine
end module

program loop_versioning_unit_slices
  use loop_versioning_unit_slices_m, only: fill_constant_scalar, &
    fill_generalized, fill_isolated_descriptors, fill_offset_slices, &
    fill_rank2_patterns, fill_sequential_owners, fill_slices
  implicit none
  real :: graph(2, 2, 3), gabor(2, 2, 3, 2)
  real :: graph_storage(4, 4, 3), gabor_storage(4, 4, 3, 2)
  real :: graph_expected(4, 4, 3), gabor_expected(4, 4, 3, 2)
  real :: rank2_values(6, 2), rank2_expected(6, 2)
  real(kind=8) :: generalized_values(4, 3, 4)
  real(kind=8) :: generalized_expected(4, 3, 4)
  real :: good_values(6, 2), good_expected(6, 2)
  real :: strided_values(6, 2), strided_expected(6, 2)
  real :: sequential_values(6, 2), sequential_expected(6, 2)
  real :: constant_values(4, 2, 2), constant_expected(4, 2, 2)
  integer :: indices(2), section_indices(2)
  character(64) :: record

  ! The facerec-shaped rank-3 and rank-4 accesses establish the baseline fast
  ! addresses for trailing scalar dimensions and two independent descriptors.
  graph = -1.0
  gabor = -1.0
  indices = [1, 3]
  record = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'
  call fill_slices(graph, gabor, indices, 2_8, record)

  if (any(graph(:, :, 1) /= reshape([1., 2., 3., 4.], [2, 2]))) &
    error stop 1
  if (any(graph(:, :, 2) /= -1.0)) error stop 2
  if (any(graph(:, :, 3) /= reshape([5., 6., 7., 8.], [2, 2]))) &
    error stop 3
  if (any(gabor(:, :, 1, 2) /= reshape([9., 10., 11., 12.], [2, 2]))) &
    error stop 4
  if (any(gabor(:, :, 2, 2) /= -1.0)) error stop 5
  if (any(gabor(:, :, 3, 2) /= &
          reshape([13., 14., 15., 16.], [2, 2]))) error stop 6
  if (any(gabor(:, :, :, 1) /= -1.0)) error stop 7

  ! The first dimension remains contiguous while the retained outer dimension
  ! has a nonstandard descriptor stride. This must use the byte fast path.
  graph_storage = -1.0
  gabor_storage = -1.0
  record = '17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32'
  call fill_slices(graph_storage(1:2, 1:4:2, :), &
                   gabor_storage(1:2, 1:4:2, :, :), indices, 2_8, record)
  graph_expected = -1.0
  graph_expected(1:2, 1:4:2, 1) = reshape([17., 18., 19., 20.], [2, 2])
  graph_expected(1:2, 1:4:2, 3) = reshape([21., 22., 23., 24.], [2, 2])
  gabor_expected = -1.0
  gabor_expected(1:2, 1:4:2, 1, 2) = &
    reshape([25., 26., 27., 28.], [2, 2])
  gabor_expected(1:2, 1:4:2, 3, 2) = &
    reshape([29., 30., 31., 32.], [2, 2])
  if (any(graph_storage /= graph_expected)) error stop 8
  if (any(gabor_storage /= gabor_expected)) error stop 9

  ! A noncontiguous first dimension makes the runtime predicate false, so the
  ! original sliced access must preserve the same semantics in the fallback.
  graph_storage = -1.0
  gabor_storage = -1.0
  record = '33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48'
  call fill_slices(graph_storage(1:4:2, 1:2, :), &
                   gabor_storage(1:4:2, 1:2, :, :), indices, 2_8, record)
  graph_expected = -1.0
  graph_expected(1:4:2, 1:2, 1) = reshape([33., 34., 35., 36.], [2, 2])
  graph_expected(1:4:2, 1:2, 3) = reshape([37., 38., 39., 40.], [2, 2])
  gabor_expected = -1.0
  gabor_expected(1:4:2, 1:2, 1, 2) = &
    reshape([41., 42., 43., 44.], [2, 2])
  gabor_expected(1:4:2, 1:2, 3, 2) = &
    reshape([45., 46., 47., 48.], [2, 2])
  if (any(graph_storage /= graph_expected)) error stop 10
  if (any(gabor_storage /= gabor_expected)) error stop 11

  ! Non-one section lower bounds exercise the retained-section correction in
  ! the byte fast path rather than relying only on structural FIR checks.
  graph_storage = -1.0
  gabor_storage = -1.0
  record = '49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64'
  call fill_offset_slices(graph_storage, gabor_storage, indices, 2_8, record)
  graph_expected = -1.0
  graph_expected(2:3, 2:3, 1) = reshape([49., 50., 51., 52.], [2, 2])
  graph_expected(2:3, 2:3, 3) = reshape([53., 54., 55., 56.], [2, 2])
  gabor_expected = -1.0
  gabor_expected(2:3, 2:3, 1, 2) = &
    reshape([57., 58., 59., 60.], [2, 2])
  gabor_expected(2:3, 2:3, 3, 2) = &
    reshape([61., 62., 63., 64.], [2, 2])
  if (any(graph_storage /= graph_expected)) error stop 12
  if (any(gabor_storage /= gabor_expected)) error stop 13

  ! Vector subscripts keep each section on direct fir.array_coor operations.
  ! Two supported owners of one descriptor use distinct dynamic lower bounds.
  rank2_values = -1.0
  rank2_expected = -1.0
  section_indices = [1, 2]
  call fill_rank2_patterns(rank2_values, 1, 4, section_indices, &
                           '65 66 67 68 69 70 71 72')
  rank2_expected(1:2, 1:2) = reshape([65., 66., 67., 68.], [2, 2])
  rank2_expected(4:5, 1:2) = reshape([69., 70., 71., 72.], [2, 2])
  if (any(rank2_values /= rank2_expected)) error stop 14

  ! Section/Section/Section validates the generalized retained-dimension
  ! formula with eight-byte elements and non-one dynamic lower bounds. The
  ! facerec procedures above cover trailing scalar coordinates separately.
  generalized_values = -1.0_8
  generalized_expected = -1.0_8
  section_indices = [1, 3]
  call fill_generalized(generalized_values, 2, 2, section_indices, &
                        '73 74 75 76 77 78 79 80')
  generalized_expected(2:3, 1, 2) = [73.0_8, 74.0_8]
  generalized_expected(2:3, 3, 2) = [75.0_8, 76.0_8]
  generalized_expected(2:3, 1, 3) = [77.0_8, 78.0_8]
  generalized_expected(2:3, 3, 3) = [79.0_8, 80.0_8]
  if (any(generalized_values /= generalized_expected)) error stop 15

  ! An unsupported dynamic-step descriptor must not disable an independent
  ! supported descriptor or be rewritten as if its step were one.
  good_values = -1.0
  good_expected = -1.0
  strided_values = -1.0
  strided_expected = -1.0
  section_indices = [1, 2]
  call fill_isolated_descriptors(good_values, strided_values, &
                                 section_indices, 2, &
                                 '81 82 83 84 85 86', &
                                 '87 88 89 90 91 92')
  good_expected(2:4, 1:2) = reshape([81., 82., 83., 84., 85., 86.], [3, 2])
  strided_expected(1:5:2, 1:2) = &
    reshape([87., 88., 89., 90., 91., 92.], [3, 2])
  if (any(good_values /= good_expected)) error stop 16
  if (any(strided_values /= strided_expected)) error stop 17

  ! One descriptor used by two sequential owners must retain the independently
  ! frozen slice facts of both loops.
  sequential_values = -1.0
  sequential_expected = -1.0
  call fill_sequential_owners(sequential_values, section_indices, &
                              '93 94 95 96', '97 98 99 100')
  sequential_expected(1:2, 1:2) = reshape([93., 94., 95., 96.], [2, 2])
  sequential_expected(4:5, 1:2) = reshape([97., 98., 99., 100.], [2, 2])
  if (any(sequential_values /= sequential_expected)) error stop 18

  ! A constant-one Scalar coordinate has a zero outer byte contribution. The
  ! fast path must retain the correct element while folding the redundant
  ! stride multiplication before later canonicalization.
  constant_values = -1.0
  constant_expected = -1.0
  section_indices = [1, 2]
  call fill_constant_scalar(constant_values, section_indices, &
                            '101 102 103 104')
  constant_expected(2:3, 1:2, 1) = &
    reshape([101.0, 102.0, 103.0, 104.0], [2, 2])
  if (any(constant_values /= constant_expected)) error stop 19

  ! A negative retained outer stride still satisfies the dimension-zero
  ! predicate and must therefore produce the same byte-fast-path addresses as
  ! the generic sliced access.
  graph_storage = -1.0
  gabor_storage = -1.0
  record = '105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120'
  call fill_slices(graph_storage(1:2, 4:2:-2, :), &
                   gabor_storage(1:2, 4:2:-2, :, :), indices, 2_8, record)
  graph_expected = -1.0
  graph_expected(1:2, 4:2:-2, 1) = &
    reshape([105., 106., 107., 108.], [2, 2])
  graph_expected(1:2, 4:2:-2, 3) = &
    reshape([109., 110., 111., 112.], [2, 2])
  gabor_expected = -1.0
  gabor_expected(1:2, 4:2:-2, 1, 2) = &
    reshape([113., 114., 115., 116.], [2, 2])
  gabor_expected(1:2, 4:2:-2, 3, 2) = &
    reshape([117., 118., 119., 120.], [2, 2])
  if (any(graph_storage /= graph_expected)) error stop 20
  if (any(gabor_storage /= gabor_expected)) error stop 21

  ! CHECK: PASS
  print '(A)', 'PASS'
end program

! Compile-time checks are paired with the executions above. They prove that
! every supported runtime helper contains the guarded byte-address fast path;
! the executable checks independently validate the addresses it computes.

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_slices(
! VERSIONED: %[[GRAPH_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[GRAPH_PRED]]
! VERSIONED: %[[GRAPH_BYTE_BOX:.*]] = fir.convert %[[GRAPH:[^ ]+]]
! VERSIONED-SAME: {{.*}}-> !fir.box<!fir.array<?xi8>>
! VERSIONED: %[[GRAPH_BASE:.*]] = fir.box_addr %[[GRAPH_BYTE_BOX]]
! VERSIONED: %[[GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[GRAPH_BASE]],
! VERSIONED-NEXT: %[[GRAPH_FAST:.*]] = fir.convert %[[GRAPH_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[GRAPH_FAST]])
! VERSIONED: %[[GABOR_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[GABOR_PRED]]
! VERSIONED: %[[GABOR_BYTE_BOX:.*]] = fir.convert %[[GABOR:[^ ]+]]
! VERSIONED-SAME: {{.*}}-> !fir.box<!fir.array<?xi8>>
! VERSIONED: %[[GABOR_BASE:.*]] = fir.box_addr %[[GABOR_BYTE_BOX]]
! VERSIONED: %[[GABOR_ADDRESS:.*]] = fir.coordinate_of %[[GABOR_BASE]],
! VERSIONED-NEXT: %[[GABOR_FAST:.*]] = fir.convert %[[GABOR_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[GABOR_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_offset_slices(
! VERSIONED: %[[OFFSET_GRAPH_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[OFFSET_GRAPH_PRED]]
! VERSIONED: %[[OFFSET_GRAPH_BYTE_BOX:.*]] = fir.convert %[[OFFSET_GRAPH:[^ ]+]]
! VERSIONED: %[[OFFSET_GRAPH_BASE:.*]] = fir.box_addr %[[OFFSET_GRAPH_BYTE_BOX]]
! VERSIONED: %[[OFFSET_GRAPH_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GRAPH_BASE]],
! VERSIONED-NEXT: %[[OFFSET_GRAPH_FAST:.*]] = fir.convert %[[OFFSET_GRAPH_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[OFFSET_GRAPH_FAST]])
! VERSIONED: %[[OFFSET_GABOR_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[OFFSET_GABOR_PRED]]
! VERSIONED: %[[OFFSET_GABOR_BYTE_BOX:.*]] = fir.convert %[[OFFSET_GABOR:[^ ]+]]
! VERSIONED: %[[OFFSET_GABOR_BASE:.*]] = fir.box_addr %[[OFFSET_GABOR_BYTE_BOX]]
! VERSIONED: %[[OFFSET_GABOR_ADDRESS:.*]] = fir.coordinate_of %[[OFFSET_GABOR_BASE]],
! VERSIONED-NEXT: %[[OFFSET_GABOR_FAST:.*]] = fir.convert %[[OFFSET_GABOR_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[OFFSET_GABOR_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_rank2_patterns(
! VERSIONED: %[[RANK2_FIRST_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[RANK2_FIRST_PRED]]
! VERSIONED: %[[RANK2_FIRST_BYTE_BOX:.*]] = fir.convert %[[RANK2:[^ ]+]]
! VERSIONED: %[[RANK2_FIRST_BASE:.*]] = fir.box_addr %[[RANK2_FIRST_BYTE_BOX]]
! VERSIONED: %[[RANK2_FIRST_ADDRESS:.*]] = fir.coordinate_of %[[RANK2_FIRST_BASE]],
! VERSIONED-NEXT: %[[RANK2_FIRST_FAST:.*]] = fir.convert %[[RANK2_FIRST_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[RANK2_FIRST_FAST]])
! VERSIONED: %[[RANK2_SECOND_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[RANK2_SECOND_PRED]]
! VERSIONED: %[[RANK2_SECOND_BYTE_BOX:.*]] = fir.convert %[[RANK2]]
! VERSIONED: %[[RANK2_SECOND_BASE:.*]] = fir.box_addr %[[RANK2_SECOND_BYTE_BOX]]
! VERSIONED: %[[RANK2_SECOND_ADDRESS:.*]] = fir.coordinate_of %[[RANK2_SECOND_BASE]],
! VERSIONED-NEXT: %[[RANK2_SECOND_FAST:.*]] = fir.convert %[[RANK2_SECOND_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[RANK2_SECOND_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_generalized(
! VERSIONED: %[[GENERAL_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[GENERAL_PRED]]
! VERSIONED: %[[GENERAL_BYTE_BOX:.*]] = fir.convert %[[GENERAL:[^ ]+]]
! VERSIONED: %[[GENERAL_BASE:.*]] = fir.box_addr %[[GENERAL_BYTE_BOX]]
! VERSIONED: %[[GENERAL_ADDRESS:.*]] = fir.coordinate_of %[[GENERAL_BASE]],
! VERSIONED-NEXT: %[[GENERAL_FAST:.*]] = fir.convert %[[GENERAL_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal64({{.*}}, %[[GENERAL_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_constant_scalar(
! VERSIONED: %[[CONSTANT_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[CONSTANT_PRED]]
! VERSIONED: %[[CONSTANT_BYTE_BOX:.*]] = fir.convert %[[CONSTANT:[^ ]+]]
! VERSIONED: %[[CONSTANT_BASE:.*]] = fir.box_addr %[[CONSTANT_BYTE_BOX]]
! VERSIONED: %[[CONSTANT_ADDRESS:.*]] = fir.coordinate_of %[[CONSTANT_BASE]],
! VERSIONED-NEXT: %[[CONSTANT_FAST:.*]] = fir.convert %[[CONSTANT_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[CONSTANT_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_isolated_descriptors(
! VERSIONED: %[[GOOD_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[GOOD_PRED]]
! VERSIONED: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD:[^ ]+]]
! VERSIONED: %[[GOOD_BASE:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
! VERSIONED: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BASE]],
! VERSIONED-NEXT: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[GOOD_FAST]])

! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_mPfill_sequential_owners(
! VERSIONED: %[[SEQUENTIAL_FIRST_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[SEQUENTIAL_FIRST_PRED]]
! VERSIONED: %[[SEQUENTIAL_FIRST_BYTE_BOX:.*]] = fir.convert %[[SEQUENTIAL:[^ ]+]]
! VERSIONED: %[[SEQUENTIAL_FIRST_BASE:.*]] = fir.box_addr %[[SEQUENTIAL_FIRST_BYTE_BOX]]
! VERSIONED: %[[SEQUENTIAL_FIRST_ADDRESS:.*]] = fir.coordinate_of %[[SEQUENTIAL_FIRST_BASE]],
! VERSIONED-NEXT: %[[SEQUENTIAL_FIRST_FAST:.*]] = fir.convert %[[SEQUENTIAL_FIRST_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[SEQUENTIAL_FIRST_FAST]])
! VERSIONED: %[[SEQUENTIAL_SECOND_PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[SEQUENTIAL_SECOND_PRED]]
! VERSIONED: %[[SEQUENTIAL_SECOND_BYTE_BOX:.*]] = fir.convert %[[SEQUENTIAL]]
! VERSIONED: %[[SEQUENTIAL_SECOND_BASE:.*]] = fir.box_addr %[[SEQUENTIAL_SECOND_BYTE_BOX]]
! VERSIONED: %[[SEQUENTIAL_SECOND_ADDRESS:.*]] = fir.coordinate_of %[[SEQUENTIAL_SECOND_BASE]],
! VERSIONED-NEXT: %[[SEQUENTIAL_SECOND_FAST:.*]] = fir.convert %[[SEQUENTIAL_SECOND_ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal{{32|64}}({{.*}}, %[[SEQUENTIAL_SECOND_FAST]])
