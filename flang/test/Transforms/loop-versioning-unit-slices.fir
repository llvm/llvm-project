// RUN: fir-opt --loop-versioning %s -o %t
// RUN: FileCheck %s --input-file=%t --enable-var-scope
// RUN: FileCheck %s --input-file=%t --check-prefix=SLICE-ARITH \
// RUN:   --enable-var-scope
// RUN: fir-opt '--loop-versioning=enable-slices=false' %s | \
// RUN:   FileCheck %s --check-prefix=DISABLED --enable-var-scope
// RUN: fir-opt --canonicalize --loop-versioning %s | \
// RUN:   FileCheck %s --check-prefix=CANONICALIZED --enable-var-scope

// Exercise accepted slice address forms, descriptor-wide rejection, owner
// selection, wrapper and type boundaries, and clone-local operands. The
// prefix-specific runs verify the transformed arithmetic, explicit disabling,
// and the canonicalized pipeline without duplicating the test IR.

module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  fir.defaultkind = "a1c4d8i4l4r4",
  fir.kindmap = "i2:1,i3:0,i16:64",
  llvm.data_layout = "e-p:64:64:64:64"
} {
  func.func private @use_i32(!fir.ref<i32>)
  func.func private @use_volatile(!fir.ref<i32, volatile>)
  func.func private @use_f32(!fir.ref<f32>)
  func.func private @use_f64(!fir.ref<f64>)
  func.func private @use_char(!fir.ref<!fir.char<1,?>>)
  func.func private @use_record(!fir.ref<!fir.type<slice_element{i:i32}>>)

  // Reduced from the facerec GraphSimFct loop. Graph(:,:,ig) and
  // GaborTrafo(:,:,x,y) are two independent slices in one owner.
  func.func @facerec_core(
      %gabor: !fir.box<!fir.array<?x?x?x?xf32>>,
      %graph: !fir.box<!fir.array<?x?x?xf32>>,
      %ig: i64, %x: i64, %y: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %graphShape = fir.shape %c8, %c8, %c8
      : (index, index, index) -> !fir.shape<3>
    %gaborShape = fir.shape %c8, %c8, %c8, %c8
      : (index, index, index, index) -> !fir.shape<4>
    %undef64 = fir.undefined i64
    %graphSlice = fir.slice %c1, %c8, %c1, %c1, %c8, %c1,
                            %ig, %undef64, %undef64
      : (index, index, index, index, index, index, i64, i64, i64)
        -> !fir.slice<3>
    %gaborSlice = fir.slice %c1, %c8, %c1, %c1, %c8, %c1,
                            %x, %undef64, %undef64,
                            %y, %undef64, %undef64
      : (index, index, index, index, index, index,
         i64, i64, i64, i64, i64, i64) -> !fir.slice<4>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %graphAddress = fir.array_coor %graph(%graphShape) [%graphSlice]
            %i, %j, %ig
          : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shape<3>, !fir.slice<3>,
             index, index, i64) -> !fir.ref<f32>
        %gaborAddress = fir.array_coor %gabor(%gaborShape) [%gaborSlice]
            %i, %j, %x, %y
          : (!fir.box<!fir.array<?x?x?x?xf32>>, !fir.shape<4>,
             !fir.slice<4>, index, index, i64, i64) -> !fir.ref<f32>
        fir.call @use_f32(%graphAddress) : (!fir.ref<f32>) -> ()
        fir.call @use_f32(%gaborAddress) : (!fir.ref<f32>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @facerec_core(
  // CHECK-SAME: %[[GABOR:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GRAPH:[^:]+]]: !fir.box
  // CHECK-SAME: %[[IG:[^:]+]]: i64
  // CHECK-SAME: %[[X:[^:]+]]: i64
  // CHECK-SAME: %[[Y:[^:]+]]: i64
  // CHECK: %[[GRAPH_SHAPE:.*]] = fir.shape
  // CHECK: %[[GABOR_SHAPE:.*]] = fir.shape
  // CHECK: %[[GRAPH_SLICE:.*]] = fir.slice
  // CHECK: %[[GABOR_SLICE:.*]] = fir.slice
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[GRAPH_DIM0:.*]] = arith.constant 0 : index
  // CHECK: %[[GRAPH_D0:.*]]:3 = fir.box_dims %[[GRAPH]], %[[GRAPH_DIM0]]
  // CHECK-NOT: fir.box_dims %[[GRAPH]],
  // CHECK: %[[GRAPH_SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[GRAPH_CMP:.*]] = arith.cmpi eq, %[[GRAPH_D0]]#2, %[[GRAPH_SIZE]]
  // CHECK: %[[GABOR_DIM0:.*]] = arith.constant 0 : index
  // CHECK: %[[GABOR_D0:.*]]:3 = fir.box_dims %[[GABOR]], %[[GABOR_DIM0]]
  // CHECK-NOT: fir.box_dims %[[GABOR]],
  // CHECK: %[[GABOR_SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[GABOR_CMP:.*]] = arith.cmpi eq, %[[GABOR_D0]]#2, %[[GABOR_SIZE]]
  // CHECK: %[[CONTIGUOUS:.*]] = arith.andi %[[GABOR_CMP]], %[[GRAPH_CMP]]
  // CHECK: fir.if %[[CONTIGUOUS]]
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[GRAPH_DIM1:.*]] = arith.constant 1 : index
  // CHECK: %[[GRAPH_D1:.*]]:3 = fir.box_dims %[[GRAPH]], %[[GRAPH_DIM1]]
  // CHECK: %[[GRAPH_DIM2:.*]] = arith.constant 2 : index
  // CHECK: %[[GRAPH_D2:.*]]:3 = fir.box_dims %[[GRAPH]], %[[GRAPH_DIM2]]
  // CHECK-NOT: fir.box_dims %[[GRAPH]],
  // CHECK: %[[GRAPH_BOX:.*]] = fir.convert %[[GRAPH]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GRAPH_BYTES:.*]] = fir.box_addr %[[GRAPH_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[GABOR_DIM1:.*]] = arith.constant 1 : index
  // CHECK: %[[GABOR_D1:.*]]:3 = fir.box_dims %[[GABOR]], %[[GABOR_DIM1]]
  // CHECK: %[[GABOR_DIM2:.*]] = arith.constant 2 : index
  // CHECK: %[[GABOR_D2:.*]]:3 = fir.box_dims %[[GABOR]], %[[GABOR_DIM2]]
  // CHECK: %[[GABOR_DIM3:.*]] = arith.constant 3 : index
  // CHECK: %[[GABOR_D3:.*]]:3 = fir.box_dims %[[GABOR]], %[[GABOR_DIM3]]
  // CHECK-NOT: fir.box_dims %[[GABOR]],
  // CHECK: %[[GABOR_BOX:.*]] = fir.convert %[[GABOR]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GABOR_BYTES:.*]] = fir.box_addr %[[GABOR_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[J:[^ ]+]] =
  // CHECK-NOT: arith.constant 0 : index
  // CHECK-NOT: arith.constant 1 : index
  // CHECK: %[[GRAPH_D0_OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK: %[[GRAPH_D0_BYTES:.*]] = arith.muli %[[GRAPH_D0_OFFSET]], %[[GRAPH_SIZE]] : index
  // CHECK: %[[GRAPH_D1_OFFSET:.*]] = arith.subi %[[J]], %[[FAST_ONE]] : index
  // CHECK: %[[GRAPH_D1_BYTES:.*]] = arith.muli %[[GRAPH_D1]]#2, %[[GRAPH_D1_OFFSET]] : index
  // CHECK: %[[GRAPH_OUTER_BYTES:.*]] = arith.addi %[[GRAPH_D0_BYTES]], %[[GRAPH_D1_BYTES]] : index
  // CHECK: %[[GRAPH_D2_INDEX:.*]] = fir.convert %[[IG]] : (i64) -> index
  // CHECK: %[[GRAPH_D2_OFFSET:.*]] = arith.subi %[[GRAPH_D2_INDEX]], %[[FAST_ONE]] : index
  // CHECK: %[[GRAPH_D2_BYTES:.*]] = arith.muli %[[GRAPH_D2]]#2, %[[GRAPH_D2_OFFSET]] : index
  // CHECK: %[[GRAPH_TOTAL:.*]] = arith.addi %[[GRAPH_OUTER_BYTES]], %[[GRAPH_D2_BYTES]] : index
  // CHECK: %[[GRAPH_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[GRAPH_BYTES]], %[[GRAPH_TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GRAPH_FAST:.*]] = fir.convert %[[GRAPH_BYTE_ADDRESS]]
  // CHECK-SAME: -> !fir.ref<f32>
  // CHECK-NOT: arith.constant 0 : index
  // CHECK-NOT: arith.constant 1 : index
  // CHECK: %[[GABOR_D0_OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK: %[[GABOR_D0_BYTES:.*]] = arith.muli %[[GABOR_D0_OFFSET]], %[[GABOR_SIZE]] : index
  // CHECK: %[[GABOR_D1_OFFSET:.*]] = arith.subi %[[J]], %[[FAST_ONE]] : index
  // CHECK: %[[GABOR_D1_BYTES:.*]] = arith.muli %[[GABOR_D1]]#2, %[[GABOR_D1_OFFSET]] : index
  // CHECK: %[[GABOR_OUTER_BYTES:.*]] = arith.addi %[[GABOR_D0_BYTES]], %[[GABOR_D1_BYTES]] : index
  // CHECK: %[[GABOR_D2_INDEX:.*]] = fir.convert %[[X]] : (i64) -> index
  // CHECK: %[[GABOR_D2_OFFSET:.*]] = arith.subi %[[GABOR_D2_INDEX]], %[[FAST_ONE]] : index
  // CHECK: %[[GABOR_D2_BYTES:.*]] = arith.muli %[[GABOR_D2]]#2, %[[GABOR_D2_OFFSET]] : index
  // CHECK: %[[GABOR_THREE_D_BYTES:.*]] = arith.addi %[[GABOR_OUTER_BYTES]], %[[GABOR_D2_BYTES]] : index
  // CHECK: %[[GABOR_D3_INDEX:.*]] = fir.convert %[[Y]] : (i64) -> index
  // CHECK: %[[GABOR_D3_OFFSET:.*]] = arith.subi %[[GABOR_D3_INDEX]], %[[FAST_ONE]] : index
  // CHECK: %[[GABOR_D3_BYTES:.*]] = arith.muli %[[GABOR_D3]]#2, %[[GABOR_D3_OFFSET]] : index
  // CHECK: %[[GABOR_TOTAL:.*]] = arith.addi %[[GABOR_THREE_D_BYTES]], %[[GABOR_D3_BYTES]] : index
  // CHECK: %[[GABOR_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[GABOR_BYTES]], %[[GABOR_TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GABOR_FAST:.*]] = fir.convert %[[GABOR_BYTE_ADDRESS]]
  // CHECK-SAME: -> !fir.ref<f32>
  // CHECK: fir.call @use_f32(%[[GRAPH_FAST]])
  // CHECK: fir.call @use_f32(%[[GABOR_FAST]])
  // CHECK: } else {
  // CHECK-NOT: fir.box_dims
  // CHECK: fir.do_loop %[[SLOW_J:[^ ]+]] =
  // CHECK: %[[GRAPH_FALLBACK:.*]] = fir.array_coor %[[GRAPH]](%[[GRAPH_SHAPE]]) [%[[GRAPH_SLICE]]] %[[I]], %[[SLOW_J]], %[[IG]]
  // CHECK: %[[GABOR_FALLBACK:.*]] = fir.array_coor %[[GABOR]](%[[GABOR_SHAPE]]) [%[[GABOR_SLICE]]] %[[I]], %[[SLOW_J]], %[[X]], %[[Y]]
  // CHECK: fir.call @use_f32(%[[GRAPH_FALLBACK]])
  // CHECK: fir.call @use_f32(%[[GABOR_FALLBACK]])
  // DISABLED-LABEL: func.func @facerec_core(
  // DISABLED-SAME: %[[GABOR:[^:]+]]: !fir.box
  // DISABLED-SAME: %[[GRAPH:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[GRAPH_ADDRESS:.*]] = fir.array_coor %[[GRAPH]]
  // DISABLED: %[[GABOR_ADDRESS:.*]] = fir.array_coor %[[GABOR]]
  // DISABLED: fir.call @use_f32(%[[GRAPH_ADDRESS]])
  // DISABLED: fir.call @use_f32(%[[GABOR_ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_f32
  // DISABLED-NOT: fir.if
  // DISABLED: return
  // SLICE-ARITH-LABEL: func.func @facerec_core(
  // SLICE-ARITH: fir.if
  // SLICE-ARITH-NOT: arith.shrsi
  // SLICE-ARITH-NOT: arith.shrui
  // SLICE-ARITH-NOT: arith.divsi
  // SLICE-ARITH-NOT: arith.divui
  // SLICE-ARITH: } else {

  // A different element type and a Section/Scalar/Section pattern demonstrate
  // that acceptance is not tied to facerec ranks, f32, or prefix sections.
  // Non-one and narrow signed section operands are widened before arithmetic.
  func.func @generalized_shape(
      %a: !fir.box<!fir.array<?x?x?xf64>>, %lower0: !fir.int<16>,
      %sliceScalar: i32, %scalar: i32, %lower2: index, %outer: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape %c8, %c8, %c8
      : (index, index, index) -> !fir.shape<3>
    %undef32 = fir.undefined i32
    %slice = fir.slice %lower0, %c8, %c1,
                       %sliceScalar, %undef32, %undef32,
                       %lower2, %c8, %c1
      : (!fir.int<16>, index, index, i32, i32, i32, index, index, index)
        -> !fir.slice<3>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %i, %scalar, %outer
        : (!fir.box<!fir.array<?x?x?xf64>>, !fir.shape<3>, !fir.slice<3>,
           index, i32, index) -> !fir.ref<f64>
      fir.call @use_f64(%address) : (!fir.ref<f64>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @generalized_shape(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[LOWER0:[^:]+]]: !fir.int<16>
  // CHECK-SAME: %[[SLICE_SCALAR:[^:]+]]: i32
  // CHECK-SAME: %[[SCALAR:[^:]+]]: i32
  // CHECK-SAME: %[[LOWER2:[^:]+]]: index
  // CHECK-SAME: %[[OUTER:[^:]+]]: index
  // CHECK: %[[SHAPE:.*]] = fir.shape
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: %[[DIM0:.*]] = arith.constant 0 : index
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]], %[[DIM0]]
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[SIZE:.*]] = arith.constant 8 : index
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM1:.*]] = arith.constant 1 : index
  // CHECK: %[[D1:.*]]:3 = fir.box_dims %[[A]], %[[DIM1]]
  // CHECK: %[[DIM2:.*]] = arith.constant 2 : index
  // CHECK: %[[D2:.*]]:3 = fir.box_dims %[[A]], %[[DIM2]]
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[A_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[A_BYTES:.*]] = fir.box_addr %[[A_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[I:[^ ]+]]
  // CHECK: %[[I_OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK: %[[LOWER0_INDEX:.*]] = fir.convert %[[LOWER0]] : (!fir.int<16>) -> index
  // CHECK: %[[LOWER0_ADJUST:.*]] = arith.subi %[[LOWER0_INDEX]], %[[FAST_ONE]] : index
  // CHECK: %[[INNER_OFFSET:.*]] = arith.addi %[[I_OFFSET]], %[[LOWER0_ADJUST]] : index
  // CHECK: %[[INNER_BYTES:.*]] = arith.muli %[[INNER_OFFSET]], %[[SIZE]] : index
  // CHECK: %[[SCALAR_INDEX:.*]] = fir.convert %[[SCALAR]] : (i32) -> index
  // CHECK: %[[SCALAR_OFFSET:.*]] = arith.subi %[[SCALAR_INDEX]], %[[FAST_ONE]] : index
  // CHECK-NOT: fir.convert %[[SLICE_SCALAR]]
  // CHECK: %[[SCALAR_BYTES:.*]] = arith.muli %[[D1]]#2, %[[SCALAR_OFFSET]] : index
  // CHECK: %[[PARTIAL:.*]] = arith.addi %[[INNER_BYTES]], %[[SCALAR_BYTES]] : index
  // CHECK: %[[OUTER_INDEX:.*]] = arith.subi %[[OUTER]], %[[FAST_ONE]] : index
  // CHECK: %[[LOWER2_ADJUST:.*]] = arith.subi %[[LOWER2]], %[[FAST_ONE]] : index
  // CHECK: %[[OUTER_OFFSET:.*]] = arith.addi %[[OUTER_INDEX]], %[[LOWER2_ADJUST]] : index
  // CHECK: %[[OUTER_BYTES:.*]] = arith.muli %[[D2]]#2, %[[OUTER_OFFSET]] : index
  // CHECK: %[[TOTAL:.*]] = arith.addi %[[PARTIAL]], %[[OUTER_BYTES]] : index
  // CHECK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[A_BYTES]], %[[TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FAST:.*]] = fir.convert %[[BYTE_ADDRESS]]
  // CHECK: fir.call @use_f64(%[[FAST]])
  // CHECK: } else {
  // CHECK: fir.do_loop %[[SLOW_I:[^ ]+]] =
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]](%[[SHAPE]]) [%[[SLICE]]] %[[SLOW_I]], %[[SCALAR]], %[[OUTER]]
  // CHECK: fir.call @use_f64(%[[FALLBACK]])
  // DISABLED-LABEL: func.func @generalized_shape(
  // DISABLED-SAME: %[[A:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: fir.call @use_f64(%[[ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_f64
  // DISABLED-NOT: fir.if
  // DISABLED: return

  // Every physical access of one accepted descriptor retains its own facts.
  // The first access is [Section, Scalar], while the second is
  // [Section, Section], so reusing or swapping their plans changes the
  // dimension-one address formula.
  func.func @two_accesses_one_descriptor(
      %a: !fir.box<!fir.array<?x?xi32>>, %scalar: index,
      %sectionLower: index, %sectionIndex: index) {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %convertedOne = fir.convert %c1_i32 : (i32) -> index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %sectionScalar = fir.slice %c1, %c8, %c1,
                               %scalar, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %sectionSection = fir.slice %convertedOne, %c8, %c1,
                                %sectionLower, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%sectionScalar] %i, %scalar
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %a [%sectionSection] %i, %sectionIndex
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @two_accesses_one_descriptor(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[SCALAR:[^:]+]]: index
  // CHECK-SAME: %[[SECTION_LOWER:[^:]+]]: index
  // CHECK-SAME: %[[SECTION_INDEX:[^:]+]]: index
  // CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
  // CHECK: %[[CONVERTED_ONE:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
  // CHECK: %[[SECTION_SCALAR:.*]] = fir.slice
  // CHECK: %[[SECTION_SECTION:.*]] = fir.slice
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2, %[[SIZE]] : index
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[D1:.*]]:3 = fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[I:[^ ]+]]
  // CHECK: %[[X_D0_OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK: %[[X_D0_BYTES:.*]] = arith.muli %[[X_D0_OFFSET]], %[[SIZE]] : index
  // CHECK: %[[X_OFFSET:.*]] = arith.subi %[[SCALAR]], %[[FAST_ONE]] : index
  // CHECK: %[[X_BYTES:.*]] = arith.muli %[[D1]]#2, %[[X_OFFSET]] : index
  // CHECK: %[[X_TOTAL:.*]] = arith.addi %[[X_D0_BYTES]], %[[X_BYTES]] : index
  // CHECK: %[[X_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[X_TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[X_REF:.*]] = fir.convert %[[X_ADDRESS]]
  // CHECK-NOT: fir.array_coor
  // CHECK: %[[Y_D0_OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK-NOT: arith.subi %[[CONVERTED_ONE]],
  // CHECK: %[[Y_D0_BYTES:.*]] = arith.muli %[[Y_D0_OFFSET]], %[[SIZE]] : index
  // CHECK: %[[Y_INDEX_OFFSET:.*]] = arith.subi %[[SECTION_INDEX]], %[[FAST_ONE]] : index
  // CHECK: %[[Y_LOWER_OFFSET:.*]] = arith.subi %[[SECTION_LOWER]], %[[FAST_ONE]] : index
  // CHECK: %[[Y_OFFSET:.*]] = arith.addi %[[Y_INDEX_OFFSET]], %[[Y_LOWER_OFFSET]] : index
  // CHECK: %[[Y_BYTES:.*]] = arith.muli %[[D1]]#2, %[[Y_OFFSET]] : index
  // CHECK: %[[Y_TOTAL:.*]] = arith.addi %[[Y_D0_BYTES]], %[[Y_BYTES]] : index
  // CHECK: %[[Y_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[Y_TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[Y_REF:.*]] = fir.convert %[[Y_ADDRESS]]
  // CHECK-NOT: fir.array_coor
  // CHECK: fir.call @use_i32(%[[X_REF]])
  // CHECK: fir.call @use_i32(%[[Y_REF]])
  // CHECK-NOT: fir.array_coor
  // CHECK: } else {
  // CHECK: %[[X_FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SECTION_SCALAR]]]
  // CHECK: %[[Y_FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SECTION_SECTION]]]
  // CHECK: fir.call @use_i32(%[[X_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[Y_FALLBACK]])
  // DISABLED-LABEL: func.func @two_accesses_one_descriptor(
  // DISABLED-SAME: %[[A:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[X_ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: %[[Y_ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: fir.call @use_i32(%[[X_ADDRESS]])
  // DISABLED: fir.call @use_i32(%[[Y_ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_i32
  // DISABLED-NOT: fir.if
  // DISABLED: return

  // Both physical accesses intentionally share one fir.slice. Preflight
  // classifies its triples once, while retaining separate coordinate checks
  // and access-local byte offsets.
  func.func @two_accesses_shared_slice(
      %a: !fir.box<!fir.array<?xi32>>, %other: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shared = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%shared] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %a [%shared] %other
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @two_accesses_shared_slice(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[OTHER:[^:]+]]: index
  // CHECK: %[[SHARED:.*]] = fir.slice
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]],
  // CHECK: %[[SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2, %[[SIZE]] : index
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[I:[^ ]+]]
  // CHECK: %[[X_OFFSET:.*]] = arith.subi %[[I]], %[[ONE]] : index
  // CHECK: %[[X_BYTES:.*]] = arith.muli %[[X_OFFSET]], %[[SIZE]] : index
  // CHECK: %[[X_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[X_BYTES]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[X_REF:.*]] = fir.convert %[[X_ADDRESS]]
  // CHECK: %[[Y_OFFSET:.*]] = arith.subi %[[OTHER]], %[[ONE]] : index
  // CHECK: %[[Y_BYTES:.*]] = arith.muli %[[Y_OFFSET]], %[[SIZE]] : index
  // CHECK: %[[Y_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[Y_BYTES]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[Y_REF:.*]] = fir.convert %[[Y_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[X_REF]])
  // CHECK: fir.call @use_i32(%[[Y_REF]])
  // CHECK: } else {
  // CHECK: %[[X_FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SHARED]]]
  // CHECK: %[[Y_FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SHARED]]]
  // CHECK: fir.call @use_i32(%[[X_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[Y_FALLBACK]])
  // DISABLED-LABEL: func.func @two_accesses_shared_slice(
  // DISABLED-SAME: %[[A:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[X_ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: %[[Y_ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: fir.call @use_i32(%[[X_ADDRESS]])
  // DISABLED: fir.call @use_i32(%[[Y_ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_i32
  // DISABLED-NOT: fir.if
  // DISABLED: return

  // A constant-one leading coordinate with a constant-one section lower has
  // zero byte offset in dimension zero. It still uses the common guarded
  // rewrite; constant folding removes only its zero address contribution.
  func.func @zero_leading_offset(
      %a: !fir.box<!fir.array<?x?xi32>>, %outer: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %slice = fir.slice %c1, %c8, %c1, %undef, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %c1, %outer
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @zero_leading_offset(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[OUTER:[^:]+]]: index
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2, %[[SIZE]] : index
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[D1:.*]]:3 = fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: fir.do_loop
  // CHECK: %[[OUTER_OFFSET:.*]] = arith.subi %[[OUTER]], %[[ONE]] : index
  // CHECK: %[[OUTER_BYTES:.*]] = arith.muli %[[D1]]#2, %[[OUTER_OFFSET]] : index
  // CHECK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[OUTER_BYTES]]
  // CHECK: %[[ADDRESS:.*]] = fir.convert %[[BYTE_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[ADDRESS]])
  // CHECK-NOT: fir.slice
  // CHECK: } else {
  // CHECK: fir.do_loop
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return
  // DISABLED-LABEL: func.func @zero_leading_offset(
  // DISABLED-SAME: %[[A:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[ADDRESS:.*]] = fir.array_coor %[[A]]
  // DISABLED: fir.call @use_i32(%[[ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_i32
  // DISABLED-NOT: fir.if
  // DISABLED: return

  // One unsupported sibling rejects the complete concrete descriptor.
  func.func @same_descriptor_unsupported_sibling(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %bad = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %a [%bad] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @same_descriptor_unsupported_sibling(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK: %[[BAD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[GOOD_ADDRESS:.*]] = fir.array_coor %[[A]]{{.*}}[%[[GOOD_SLICE]]]
  // CHECK: %[[BAD_ADDRESS:.*]] = fir.array_coor %[[A]]{{.*}}[%[[BAD_SLICE]]]
  // CHECK: fir.call @use_i32(%[[GOOD_ADDRESS]])
  // CHECK: fir.call @use_i32(%[[BAD_ADDRESS]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // A flat sibling of either indexing operation kind is outside the complete
  // sliced-descriptor contract, regardless of which access is discovered
  // first.
  func.func @mixed_flat_and_slice_rejected(
      %flatFirst: !fir.box<!fir.array<?xi32>>,
      %sliceFirst: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %flatA = fir.array_coor %flatFirst %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      %sliceA = fir.array_coor %flatFirst [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %sliceB = fir.array_coor %sliceFirst [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %flatB = fir.coordinate_of %sliceFirst, %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%flatA) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%sliceA) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%sliceB) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%flatB) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @mixed_flat_and_slice_rejected(
  // CHECK-SAME: %[[FLAT_FIRST:[^:]+]]: !fir.box
  // CHECK-SAME: %[[SLICE_FIRST:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:[^ ]+]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[FLAT_FIRST]],
  // CHECK-NOT: fir.box_dims %[[SLICE_FIRST]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[FLAT_A:[^ ]+]] = fir.array_coor %[[FLAT_FIRST]] %[[I]]
  // CHECK: %[[SLICE_A:[^ ]+]] = fir.array_coor %[[FLAT_FIRST]] [%[[SLICE]]] %[[I]]
  // CHECK: %[[SLICE_B:[^ ]+]] = fir.array_coor %[[SLICE_FIRST]] [%[[SLICE]]] %[[I]]
  // CHECK: %[[FLAT_B:[^ ]+]] = fir.coordinate_of %[[SLICE_FIRST]], %[[I]]
  // CHECK: fir.call @use_i32(%[[FLAT_A]])
  // CHECK: fir.call @use_i32(%[[SLICE_A]])
  // CHECK: fir.call @use_i32(%[[SLICE_B]])
  // CHECK: fir.call @use_i32(%[[FLAT_B]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // A flat sibling owner leaves no complete sliced-owner set. The
  // sliced owner stays generic, while the independent flat owner may retain
  // the existing slice-free versioning path.
  func.func @sibling_flat_owner_rejects_slices(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %sliced = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%sliced) : (!fir.ref<i32>) -> ()
    }
    fir.do_loop %j = %c1 to %c8 step %c1 {
      %flat = fir.array_coor %a %j
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%flat) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @sibling_flat_owner_rejects_slices(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[SLICED:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]],
  // CHECK: %[[SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2, %[[SIZE]] : index
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[FLAT_BOX:.*]] = fir.convert %[[A]]
  // CHECK: %[[FLAT_BASE:.*]] = fir.box_addr %[[FLAT_BOX]]
  // CHECK: %[[FLAT_ADDRESS:.*]] = fir.coordinate_of %[[FLAT_BASE]],
  // CHECK: fir.call @use_i32(%[[FLAT_ADDRESS]])
  // CHECK: } else {
  // CHECK: %[[FLAT_FALLBACK:.*]] = fir.array_coor %[[A]]
  // CHECK: fir.call @use_i32(%[[FLAT_FALLBACK]])
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: return

  // Descriptor-wide rejection uses the concrete wrapper identity. A sliced
  // owner and a slice-free owner of the same declared descriptor must not let
  // normalization hide the conflict; only the slice-free owner is versioned.
  func.func @wrapped_flat_owner_rejects_slices(
      %root: !fir.box<!fir.array<?xi32>>) {
    %declared = fir.declare %root {uniq_name = "wrapped"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %sliced = fir.array_coor %declared [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%sliced) : (!fir.ref<i32>) -> ()
    }
    fir.do_loop %j = %c1 to %c8 step %c1 {
      %flat = fir.array_coor %declared %j
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%flat) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @wrapped_flat_owner_rejects_slices(
  // CHECK-SAME: %[[ROOT:[^:]+]]: !fir.box
  // CHECK: %[[DECLARED:.*]] = fir.declare %[[ROOT]]
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[SLICED:.*]] = fir.array_coor %[[DECLARED]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[SLICED]])
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[DECLARED]],
  // CHECK: %[[SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2, %[[SIZE]] : index
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[FLAT_BOX:.*]] = fir.convert %[[DECLARED]]
  // CHECK: %[[FLAT_BASE:.*]] = fir.box_addr %[[FLAT_BOX]]
  // CHECK: %[[FLAT_ADDRESS:.*]] = fir.coordinate_of %[[FLAT_BASE]],
  // CHECK: fir.call @use_i32(%[[FLAT_ADDRESS]])
  // CHECK: } else {
  // CHECK: %[[FLAT_FALLBACK:.*]] = fir.array_coor %[[DECLARED]]
  // CHECK: fir.call @use_i32(%[[FLAT_FALLBACK]])
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: return

  // A sliced descriptor may be published only when every owner is entirely
  // slice-based. The second owner also contains independent slice-free work,
  // so both sliced uses of %a remain generic while %b is still versioned.
  func.func @publication_rejects_incomplete_owner_set(
      %a: !fir.box<!fir.array<?xi32>>,
      %b: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %first = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%first) : (!fir.ref<i32>) -> ()
    }
    fir.do_loop %j = %c1 to %c8 step %c1 {
      %second = fir.array_coor %a [%slice] %j
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %flat = fir.array_coor %b %j
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%second) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%flat) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @publication_rejects_incomplete_owner_set(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[B:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[FIRST:.*]] = fir.array_coor %[[A]] [%[[SLICE]]] %[[I]]
  // CHECK: fir.call @use_i32(%[[FIRST]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: fir.box_dims %[[B]],
  // CHECK: fir.if
  // CHECK: %[[B_BOX:.*]] = fir.convert %[[B]]
  // CHECK: %[[B_BASE:.*]] = fir.box_addr %[[B_BOX]]
  // CHECK: fir.do_loop %[[FAST_J:[^ ]+]] =
  // CHECK: %[[SECOND_FAST:.*]] = fir.array_coor %[[A]] [%[[SLICE]]] %[[FAST_J]]
  // CHECK: %[[B_FAST:.*]] = fir.coordinate_of %[[B_BASE]],
  // CHECK: fir.call @use_i32(%[[SECOND_FAST]])
  // CHECK: fir.call @use_i32(%[[B_FAST]])
  // CHECK: } else {
  // CHECK: fir.do_loop %[[FALLBACK_J:[^ ]+]] =
  // CHECK: %[[SECOND_FALLBACK:.*]] = fir.array_coor %[[A]] [%[[SLICE]]] %[[FALLBACK_J]]
  // CHECK: %[[B_FALLBACK:.*]] = fir.array_coor %[[B]] %[[FALLBACK_J]]
  // CHECK: fir.call @use_i32(%[[SECOND_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[B_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: return

  // Rejection is descriptor-local: an independent unit slice remains
  // versionable in the same owner.
  func.func @descriptor_independence(
      %badArg: !fir.box<!fir.array<?xi32>>,
      %goodArg: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %bad = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %badArg [%bad] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %goodArg [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @descriptor_independence(
  // CHECK-SAME: %[[BAD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[BAD_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: %[[GOOD_D0:.*]]:3 = fir.box_dims %[[GOOD]],
  // CHECK: %[[GOOD_SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[GOOD_PRED:.*]] = arith.cmpi eq, %[[GOOD_D0]]#2, %[[GOOD_SIZE]] : index
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.if %[[GOOD_PRED]]
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[BAD]]{{.*}}[%[[BAD_SLICE]]]
  // CHECK: fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[BAD]]{{.*}}[%[[BAD_SLICE]]]
  // CHECK: fir.array_coor %[[GOOD]]{{.*}}[%[[GOOD_SLICE]]]
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: return

  // Canonicalization folds the explicit i1-to-i8 zero extension to positive
  // one, making both it and the ordinary unit slice eligible.
  // CANONICALIZED-LABEL: func.func @i1_operands_rejected(
  // CANONICALIZED-SAME: %[[BAD_STEP:[^:]+]]: !fir.box
  // CANONICALIZED-SAME: %[[BAD_INDEX:[^:]+]]: !fir.box
  // CANONICALIZED-SAME: %[[BAD_LOWER:[^:]+]]: !fir.box
  // CANONICALIZED-SAME: %[[CONVERTED_STEP:[^:]+]]: !fir.box
  // CANONICALIZED-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_STEP]],
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_INDEX]],
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_LOWER]],
  // CANONICALIZED: fir.box_dims %[[CONVERTED_STEP]],
  // CANONICALIZED: fir.box_dims %[[GOOD]],
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_STEP]],
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_INDEX]],
  // CANONICALIZED-NOT: fir.box_dims %[[BAD_LOWER]],
  // CANONICALIZED: return

  // Reject raw i1 values in every consumed operand position. Without
  // canonicalization, also conservatively reject the explicit i1-to-i8-to-index
  // chain. An independent descriptor with an ordinary unit step must remain
  // eligible.
  func.func @i1_operands_rejected(
      %badStepArg: !fir.box<!fir.array<?xi32>>,
      %badIndexArg: !fir.box<!fir.array<?xi32>>,
      %badLowerArg: !fir.box<!fir.array<?xi32>>,
      %convertedStepArg: !fir.box<!fir.array<?xi32>>,
      %goodArg: !fir.box<!fir.array<?xi32>>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %wideTrue = fir.convert %true : (i1) -> i8
    %convertedUnit = fir.convert %wideTrue : (i8) -> index
    %c8 = arith.constant 8 : index
    %badStep = fir.slice %c1, %c8, %true
      : (index, index, i1) -> !fir.slice<1>
    %badLower = fir.slice %true, %c8, %c1
      : (i1, index, index) -> !fir.slice<1>
    %converted = fir.slice %c1, %c8, %convertedUnit
      : (index, index, index) -> !fir.slice<1>
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %badStepAddress = fir.array_coor %badStepArg [%badStep] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %badIndexAddress = fir.array_coor %badIndexArg [%good] %true
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i1)
          -> !fir.ref<i32>
      %badLowerAddress = fir.array_coor %badLowerArg [%badLower] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %convertedStepAddress = fir.array_coor %convertedStepArg [%converted] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %goodArg [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%badStepAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%badIndexAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%badLowerAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%convertedStepAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @i1_operands_rejected(
  // CHECK-SAME: %[[BAD_STEP:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD_INDEX:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD_LOWER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[CONVERTED_STEP:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[WIDE_TRUE:.*]] = fir.convert %[[TRUE]] : (i1) -> i8
  // CHECK: %[[UNIT:.*]] = fir.convert %[[WIDE_TRUE]] : (i8) -> index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[BAD_STEP_SLICE:.*]] = fir.slice %[[C1]], %[[C8]], %[[TRUE]]
  // CHECK: %[[BAD_LOWER_SLICE:.*]] = fir.slice %[[TRUE]], %[[C8]], %[[C1]]
  // CHECK: %[[CONVERTED_SLICE:.*]] = fir.slice %[[C1]], %[[C8]], %[[UNIT]]
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice %[[C1]], %[[C8]], %[[C1]]
  // CHECK-NOT: fir.box_dims %[[BAD_STEP]],
  // CHECK-NOT: fir.box_dims %[[BAD_INDEX]],
  // CHECK-NOT: fir.box_dims %[[BAD_LOWER]],
  // CHECK-NOT: fir.box_dims %[[CONVERTED_STEP]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD_STEP]],
  // CHECK-NOT: fir.box_dims %[[BAD_INDEX]],
  // CHECK-NOT: fir.box_dims %[[BAD_LOWER]],
  // CHECK-NOT: fir.box_dims %[[CONVERTED_STEP]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[FAST_I:[^ ]+]] =
  // CHECK: %[[BAD_STEP_FAST:.*]] = fir.array_coor %[[BAD_STEP]] [%[[BAD_STEP_SLICE]]] %[[FAST_I]]
  // CHECK: %[[BAD_INDEX_FAST:.*]] = fir.array_coor %[[BAD_INDEX]] [%[[GOOD_SLICE]]] %[[TRUE]]
  // CHECK: %[[BAD_LOWER_FAST:.*]] = fir.array_coor %[[BAD_LOWER]] [%[[BAD_LOWER_SLICE]]] %[[FAST_I]]
  // CHECK: %[[CONVERTED_FAST:.*]] = fir.array_coor %[[CONVERTED_STEP]] [%[[CONVERTED_SLICE]]] %[[FAST_I]]
  // CHECK-NOT: fir.array_coor %[[GOOD]]
  // CHECK: %[[GOOD_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_BYTE_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[BAD_STEP_FAST]])
  // CHECK: fir.call @use_i32(%[[BAD_INDEX_FAST]])
  // CHECK: fir.call @use_i32(%[[BAD_LOWER_FAST]])
  // CHECK: fir.call @use_i32(%[[CONVERTED_FAST]])
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK: } else {
  // CHECK: fir.do_loop %[[SLOW_I:[^ ]+]] =
  // CHECK: %[[BAD_STEP_FALLBACK:.*]] = fir.array_coor %[[BAD_STEP]] [%[[BAD_STEP_SLICE]]] %[[SLOW_I]]
  // CHECK: %[[BAD_INDEX_FALLBACK:.*]] = fir.array_coor %[[BAD_INDEX]] [%[[GOOD_SLICE]]] %[[TRUE]]
  // CHECK: %[[BAD_LOWER_FALLBACK:.*]] = fir.array_coor %[[BAD_LOWER]] [%[[BAD_LOWER_SLICE]]] %[[SLOW_I]]
  // CHECK: %[[CONVERTED_FALLBACK:.*]] = fir.array_coor %[[CONVERTED_STEP]] [%[[CONVERTED_SLICE]]] %[[SLOW_I]]
  // CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD]] [%[[GOOD_SLICE]]] %[[SLOW_I]]
  // CHECK: fir.call @use_i32(%[[BAD_STEP_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[BAD_INDEX_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[BAD_LOWER_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[CONVERTED_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[GOOD_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[BAD_STEP]],
  // CHECK-NOT: fir.box_dims %[[BAD_INDEX]],
  // CHECK-NOT: fir.box_dims %[[BAD_LOWER]],
  // CHECK-NOT: fir.box_dims %[[CONVERTED_STEP]],
  // CHECK: return

  // A direct i1-to-index conversion sign-extends the set bit to -1, so it must
  // remain ineligible even when canonicalization materializes that constant.
  func.func @direct_i1_to_index_step_rejected(
      %arg: !fir.box<!fir.array<?xi32>>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %step = fir.convert %true : (i1) -> index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %arg [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @direct_i1_to_index_step_rejected(
  // CHECK-SAME: %[[ARG:[^:]+]]: !fir.box
  // CHECK: %[[STEP:.*]] = fir.convert %{{.*}} : (i1) -> index
  // CHECK: %[[SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %[[STEP]]
  // CHECK-NOT: fir.box_dims %[[ARG]],
  // CHECK: fir.do_loop
  // CHECK: fir.array_coor %[[ARG]] [%[[SLICE]]]
  // CHECK-NOT: fir.box_dims %[[ARG]],
  // CHECK: return
  // CANONICALIZED-LABEL: func.func @direct_i1_to_index_step_rejected(
  // CANONICALIZED-SAME: %[[ARG:[^:]+]]: !fir.box
  // CANONICALIZED: %[[STEP:.*]] = arith.constant -1 : index
  // CANONICALIZED: %[[SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %[[STEP]]
  // CANONICALIZED-NOT: fir.box_dims %[[ARG]],
  // CANONICALIZED: fir.do_loop
  // CANONICALIZED: fir.array_coor %[[ARG]] [%[[SLICE]]]
  // CANONICALIZED-NOT: fir.box_dims %[[ARG]],
  // CANONICALIZED: return

  // A signed one-bit intermediate changes a positive one to -1 when it is
  // widened. Preserve the section-lower adjustment instead of looking through
  // such a conversion chain and eliding it.
  func.func @signed_one_bit_convert_adjusted(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1_i32 = arith.constant 1 : i32
    %signedBit = fir.convert %c1_i32 : (i32) -> si1
    %lower = fir.convert %signedBit : (si1) -> i64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %lower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @signed_one_bit_convert_adjusted(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[ONE_I32:.*]] = arith.constant 1 : i32
  // CHECK: %[[SIGNED_BIT:.*]] = fir.convert %[[ONE_I32]] : (i32) -> si1
  // CHECK: %[[LOWER:.*]] = fir.convert %[[SIGNED_BIT]] : (si1) -> i64
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[LOWER_INDEX:.*]] = fir.convert %[[LOWER]] : (i64) -> index
  // CHECK: %[[LOWER_ADJUST:.*]] = arith.subi %[[LOWER_INDEX]], %[[FAST_ONE]] : index
  // CHECK: arith.addi {{.*}}, %[[LOWER_ADJUST]] : index
  // CHECK: fir.coordinate_of %[[BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[A]]

  // FIR integer kinds are signed. With this kind mapping, !fir.int<2> lowers
  // to i1, so widening its set bit produces -1. Preserve the section-lower
  // adjustment just as for the builtin signed one-bit form above.
  func.func @kind_mapped_one_bit_lower_adjusted(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1_i32 = arith.constant 1 : i32
    %kindBit = fir.convert %c1_i32 : (i32) -> !fir.int<2>
    %lower = fir.convert %kindBit : (!fir.int<2>) -> i64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %lower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @kind_mapped_one_bit_lower_adjusted(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[ONE_I32:.*]] = arith.constant 1 : i32
  // CHECK: %[[KIND_BIT:.*]] = fir.convert %[[ONE_I32]] : (i32) -> !fir.int<2>
  // CHECK: %[[LOWER:.*]] = fir.convert %[[KIND_BIT]] : (!fir.int<2>) -> i64
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[LOWER_INDEX:.*]] = fir.convert %[[LOWER]] : (i64) -> index
  // CHECK: %[[LOWER_ADJUST:.*]] = arith.subi %[[LOWER_INDEX]], %[[FAST_ONE]] : index
  // CHECK: arith.addi {{.*}}, %[[LOWER_ADJUST]] : index
  // CHECK: fir.coordinate_of %[[BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[A]]

  // Proven source lowers equal to one do not participate in fast-path
  // arithmetic, so their original integer widths do not restrict versioning.
  func.func @static_one_lowers_ignore_source_width(
      %narrowArg: !fir.box<!fir.array<?xi32>>,
      %wideArg: !fir.box<!fir.array<?xi32>>) {
    %narrowSource = arith.constant 1 : i16
    %narrowLower = fir.convert %narrowSource : (i16) -> ui16
    %wideLower = arith.constant 1 : i128
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %narrowSlice = fir.slice %narrowLower, %c8, %c1
      : (ui16, index, index) -> !fir.slice<1>
    %wideSlice = fir.slice %wideLower, %c8, %c1
      : (i128, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %narrowAddress = fir.array_coor %narrowArg [%narrowSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %wideAddress = fir.array_coor %wideArg [%wideSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%narrowAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%wideAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @static_one_lowers_ignore_source_width(
  // CHECK-SAME: %[[NARROW_ARG:[^:]+]]: !fir.box
  // CHECK-SAME: %[[WIDE_ARG:[^:]+]]: !fir.box
  // CHECK: %[[NARROW_SOURCE:.*]] = arith.constant 1 : i16
  // CHECK: %[[NARROW_LOWER:.*]] = fir.convert %[[NARROW_SOURCE]] : (i16) -> ui16
  // CHECK: %[[WIDE_LOWER:.*]] = arith.constant 1 : i128
  // CHECK: %[[NARROW_SLICE:.*]] = fir.slice %[[NARROW_LOWER]],
  // CHECK: %[[WIDE_SLICE:.*]] = fir.slice %[[WIDE_LOWER]],
  // CHECK: fir.box_dims %[[NARROW_ARG]],
  // CHECK: fir.box_dims %[[WIDE_ARG]],
  // CHECK: fir.if
  // CHECK: %[[NARROW_BYTE_BOX:.*]] = fir.convert %[[NARROW_ARG]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[NARROW_BYTES:.*]] = fir.box_addr %[[NARROW_BYTE_BOX]]
  // CHECK: %[[WIDE_BYTE_BOX:.*]] = fir.convert %[[WIDE_ARG]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[WIDE_BYTES:.*]] = fir.box_addr %[[WIDE_BYTE_BOX]]
  // CHECK: fir.do_loop
  // CHECK-NOT: fir.convert %[[NARROW_LOWER]]
  // CHECK-NOT: fir.convert %[[WIDE_LOWER]]
  // CHECK-NOT: fir.array_coor %[[NARROW_ARG]]
  // CHECK-NOT: fir.array_coor %[[WIDE_ARG]]
  // CHECK: %[[NARROW_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[NARROW_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[NARROW_FAST:.*]] = fir.convert %[[NARROW_BYTE_ADDRESS]]
  // CHECK: %[[WIDE_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[WIDE_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[WIDE_FAST:.*]] = fir.convert %[[WIDE_BYTE_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[NARROW_FAST]])
  // CHECK-NEXT: fir.call @use_i32(%[[WIDE_FAST]])
  // CHECK: } else {
  // CHECK: %[[NARROW_FALLBACK:.*]] = fir.array_coor %[[NARROW_ARG]] [%[[NARROW_SLICE]]]
  // CHECK: %[[WIDE_FALLBACK:.*]] = fir.array_coor %[[WIDE_ARG]] [%[[WIDE_SLICE]]]
  // CHECK: fir.call @use_i32(%[[NARROW_FALLBACK]])
  // CHECK-NEXT: fir.call @use_i32(%[[WIDE_FALLBACK]])
  // CHECK: return

  // A proven unit lower is irrelevant to address materialization, but it must
  // not hide a non-unit section step. Reject only that descriptor while an
  // independent unit-step descriptor in the same owner remains eligible.
  func.func @static_one_lower_does_not_hide_nonunit_step(
      %bad: !fir.box<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %wideOne = arith.constant 1 : i128
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %badSlice = fir.slice %wideOne, %c8, %c2
      : (i128, index, index) -> !fir.slice<1>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %badAddress = fir.array_coor %bad [%badSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%badAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @static_one_lower_does_not_hide_nonunit_step(
  // CHECK-SAME: %[[BAD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[BAD_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.if
  // CHECK: fir.array_coor %[[BAD]]{{.*}}[%[[BAD_SLICE]]]
  // CHECK: fir.coordinate_of
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[BAD]]{{.*}}[%[[BAD_SLICE]]]
  // CHECK: fir.array_coor %[[GOOD]]{{.*}}[%[[GOOD_SLICE]]]
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: return

  // Non-unit lower bounds still require a supported address representation.
  // Reject a too-wide signed value, a narrow unsigned value, and a FIR integer
  // kind with no mapped width without affecting an independent descriptor.
  func.func @unsupported_section_lowers_remain_rejected(
      %wide: !fir.box<!fir.array<?xi32>>,
      %unsigned: !fir.box<!fir.array<?xi32>>,
      %zeroWidth: !fir.box<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %wideLower = arith.constant 2 : i128
    %unsignedSource = arith.constant 5 : i16
    %unsignedLower = fir.convert %unsignedSource : (i16) -> ui16
    %zeroSource = arith.constant 2 : i32
    %zeroLower = fir.convert %zeroSource : (i32) -> !fir.int<3>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %wideSlice = fir.slice %wideLower, %c8, %c1
      : (i128, index, index) -> !fir.slice<1>
    %unsignedSlice = fir.slice %unsignedLower, %c8, %c1
      : (ui16, index, index) -> !fir.slice<1>
    %zeroSlice = fir.slice %zeroLower, %c8, %c1
      : (!fir.int<3>, index, index) -> !fir.slice<1>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %wideAddress = fir.array_coor %wide [%wideSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %unsignedAddress = fir.array_coor %unsigned [%unsignedSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %zeroAddress = fir.array_coor %zeroWidth [%zeroSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%wideAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%unsignedAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%zeroAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @unsupported_section_lowers_remain_rejected(
  // CHECK-SAME: %[[WIDE:[^:]+]]: !fir.box
  // CHECK-SAME: %[[UNSIGNED:[^:]+]]: !fir.box
  // CHECK-SAME: %[[ZERO:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[WIDE_SLICE:.*]] = fir.slice
  // CHECK: %[[UNSIGNED_SLICE:.*]] = fir.slice
  // CHECK: %[[ZERO_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[WIDE]],
  // CHECK-NOT: fir.box_dims %[[UNSIGNED]],
  // CHECK-NOT: fir.box_dims %[[ZERO]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[WIDE]],
  // CHECK-NOT: fir.box_dims %[[UNSIGNED]],
  // CHECK-NOT: fir.box_dims %[[ZERO]],
  // CHECK: fir.if
  // CHECK: fir.array_coor %[[WIDE]]{{.*}}[%[[WIDE_SLICE]]]
  // CHECK: fir.array_coor %[[UNSIGNED]]{{.*}}[%[[UNSIGNED_SLICE]]]
  // CHECK: fir.array_coor %[[ZERO]]{{.*}}[%[[ZERO_SLICE]]]
  // CHECK: fir.coordinate_of
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[WIDE]]{{.*}}[%[[WIDE_SLICE]]]
  // CHECK: fir.array_coor %[[UNSIGNED]]{{.*}}[%[[UNSIGNED_SLICE]]]
  // CHECK: fir.array_coor %[[ZERO]]{{.*}}[%[[ZERO_SLICE]]]
  // CHECK: fir.array_coor %[[GOOD]]{{.*}}[%[[GOOD_SLICE]]]
  // CHECK: return

  // A one-bit unsigned intermediate zero-extends to index. The resulting
  // section lower is one, so it must not add a lower-bound adjustment.
  func.func @converted_ui1_lower_is_one(
      %a: !fir.box<!fir.array<?xi32>>) {
    %source = arith.constant 1 : i8
    %bit = fir.convert %source : (i8) -> ui1
    %lower = fir.convert %bit : (ui1) -> index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %lower, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @converted_ui1_lower_is_one(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SOURCE:.*]] = arith.constant 1 : i8
  // CHECK: %[[BIT:.*]] = fir.convert %[[SOURCE]] : (i8) -> ui1
  // CHECK: %[[LOWER:.*]] = fir.convert %[[BIT]] : (ui1) -> index
  // CHECK: %[[SLICE:.*]] = fir.slice %[[LOWER]],
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK-NOT: arith.subi %[[LOWER]],
  // CHECK: %[[OFFSET:.*]] = arith.subi %[[I]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK: %[[FAST:.*]] = fir.convert %[[BYTE_ADDRESS]]
  // CHECK-NEXT: fir.call @use_i32(%[[FAST]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]] [%[[SLICE]]]
  // CHECK-NEXT: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // The same effective one-bit value cannot be accepted as a positive unit
  // step. Keep this descriptor on the generic path.
  func.func @kind_mapped_one_bit_step_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1_i32 = arith.constant 1 : i32
    %kindBit = fir.convert %c1_i32 : (i32) -> !fir.int<2>
    %step = fir.convert %kindBit : (!fir.int<2>) -> i64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, i64) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @kind_mapped_one_bit_step_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: return

  // A truncating conversion may produce a positive one even when its source
  // constant is not one. Classify the final bit pattern and accept the slice.
  func.func @truncated_static_one_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c257 = arith.constant 257 : i16
    %step = fir.convert %c257 : (i16) -> i8
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, i8) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @truncated_static_one_step(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SOURCE:.*]] = arith.constant 257 : i16
  // CHECK: %[[STEP:.*]] = fir.convert %[[SOURCE]] : (i16) -> i8
  // CHECK: %[[SLICE:.*]] = fir.slice {{.*}}%[[STEP]]
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[FAST:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[TYPED:.*]] = fir.convert %[[FAST]] : (!fir.ref<i8>) -> !fir.ref<i32>
  // CHECK: fir.call @use_i32(%[[TYPED]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // A non-index constant one is already a unit step without a conversion.
  func.func @direct_i8_static_one_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %step = arith.constant 1 : i8
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, i8) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @direct_i8_static_one_step(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: %[[FAST:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK: %[[TYPED:.*]] = fir.convert %[[FAST]] : (!fir.ref<i8>) -> !fir.ref<i32>
  // CHECK: fir.call @use_i32(%[[TYPED]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // Signedness after a truncation determines how the resulting one is widened.
  func.func @unsigned_after_trunc_static_one_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c257 = arith.constant 257 : i16
    %step = fir.convert %c257 : (i16) -> ui8
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, ui8) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @unsigned_after_trunc_static_one_step(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: %[[FAST:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK: %[[TYPED:.*]] = fir.convert %[[FAST]] : (!fir.ref<i8>) -> !fir.ref<i32>
  // CHECK: fir.call @use_i32(%[[TYPED]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // Preserve every conversion in a chain whose intermediate value is index.
  func.func @index_middle_static_one_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c257 = arith.constant 257 : i16
    %narrow = fir.convert %c257 : (i16) -> i8
    %asIndex = fir.convert %narrow : (i8) -> index
    %step = fir.convert %asIndex : (index) -> i64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, i64) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @index_middle_static_one_step(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: %[[FAST:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK: %[[TYPED:.*]] = fir.convert %[[FAST]] : (!fir.ref<i8>) -> !fir.ref<i32>
  // CHECK: fir.call @use_i32(%[[TYPED]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // Generic XArrayCoor truncates a wider step to its 64-bit address index.
  func.func @wide_static_one_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %wide = arith.constant 18446744073709551617 : i128
    %middle = fir.convert %wide : (i128) -> i96
    %narrow = fir.convert %middle : (i96) -> i32
    %step = fir.convert %narrow : (i32) -> i128
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, i128) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @wide_static_one_step(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK: %[[FAST:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK: %[[TYPED:.*]] = fir.convert %[[FAST]] : (!fir.ref<i8>) -> !fir.ref<i32>
  // CHECK: fir.call @use_i32(%[[TYPED]])
  // CHECK: } else {
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // A conversion from a non-integer value is not an integer bit-cast chain.
  func.func @noninteger_static_one_step_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %one = arith.constant 1.0 : f32
    %step = fir.convert %one : (f32) -> index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @noninteger_static_one_step_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: %[[ADDRESS:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[ADDRESS]])
  // CHECK: return

  // A kind mapped to zero bits cannot participate in address arithmetic.
  func.func @zero_kind_width_static_one_step_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1_i32 = arith.constant 1 : i32
    %kindZero = fir.convert %c1_i32 : (i32) -> !fir.int<3>
    %step = fir.convert %kindZero : (!fir.int<3>) -> index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %step
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @zero_kind_width_static_one_step_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: %[[ADDRESS:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[ADDRESS]])
  // CHECK: return

  // Mixed owners are outside the initial slice scope. The independent
  // flat descriptor keeps the slice-free path, while the slice remains
  // generic.
  func.func @slice_with_independent_flat_uses_slice_free_path(
      %slicedArg: !fir.box<!fir.array<?xi32>>,
      %flatArg: !fir.box<!fir.array<?x?xi32>>, %outer: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %flat = fir.array_coor %flatArg %i, %outer
        : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
      %sliced = fir.array_coor %slicedArg [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%sliced) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%flat) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @slice_with_independent_flat_uses_slice_free_path(
  // CHECK-SAME: %[[SLICED:[^:]+]]: !fir.box
  // CHECK-SAME: %[[FLAT:[^:]+]]: !fir.box
  // CHECK-NOT: fir.box_dims %[[SLICED]],
  // CHECK: fir.box_dims %[[FLAT]],
  // CHECK-NOT: fir.box_dims %[[SLICED]],
  // CHECK: fir.if
  // CHECK: %[[FLAT_TYPED:.*]] = fir.box_addr
  // CHECK-SAME: !fir.ref<!fir.array<?xi32>>
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop
  // CHECK: arith.muli
  // CHECK: arith.shrsi
  // CHECK: arith.addi
  // CHECK: %[[FLAT_FAST:.*]] = fir.coordinate_of %[[FLAT_TYPED]],
  // CHECK: %[[SLICED_FAST:.*]] = fir.array_coor %[[SLICED]]{{.*}}[%{{.*}}]
  // CHECK: fir.call @use_i32(%[[SLICED_FAST]])
  // CHECK: fir.call @use_i32(%[[FLAT_FAST]])
  // CHECK: } else {
  // CHECK: %[[FLAT_SLOW:.*]] = fir.array_coor %[[FLAT]]
  // CHECK: %[[SLICED_SLOW:.*]] = fir.array_coor %[[SLICED]]{{.*}}[%{{.*}}]
  // CHECK: fir.call @use_i32(%[[SLICED_SLOW]])
  // CHECK: fir.call @use_i32(%[[FLAT_SLOW]])

  // DISABLED-LABEL: func.func @slice_with_independent_flat_uses_slice_free_path(
  // DISABLED-SAME: %[[SLICED:[^:]+]]: !fir.box
  // DISABLED-SAME: %[[FLAT:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.box_dims %[[SLICED]],
  // DISABLED: fir.box_dims %[[FLAT]],
  // DISABLED-NOT: fir.box_dims %[[SLICED]],
  // DISABLED: fir.if
  // DISABLED: %[[FLAT_TYPED:.*]] = fir.box_addr
  // DISABLED: fir.do_loop
  // DISABLED: %[[FLAT_FAST:.*]] = fir.coordinate_of %[[FLAT_TYPED]],
  // DISABLED: %[[SLICED_FAST:.*]] = fir.array_coor %[[SLICED]]
  // DISABLED: fir.call @use_i32(%[[SLICED_FAST]])
  // DISABLED: fir.call @use_i32(%[[FLAT_FAST]])
  // DISABLED: } else {
  // DISABLED: %[[FLAT_SLOW:.*]] = fir.array_coor %[[FLAT]]
  // DISABLED: %[[SLICED_SLOW:.*]] = fir.array_coor %[[SLICED]]
  // DISABLED: fir.call @use_i32(%[[SLICED_SLOW]])
  // DISABLED: fir.call @use_i32(%[[FLAT_SLOW]])
  // DISABLED: return

  // A provisional slice-free candidate may be removed after an inner sliced
  // use rejects that descriptor. Preserve an independent sliced descriptor
  // until final owner selection instead of losing both transformations.
  func.func @late_slice_free_rejection_preserves_slice(
      %slicedArg: !fir.box<!fir.array<?xi32>>,
      %conflictedArg: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slicedSection = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %conflictedSection = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %sliced = fir.array_coor %slicedArg [%slicedSection] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %flat = fir.array_coor %conflictedArg %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%sliced) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%flat) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %conflicted = fir.array_coor %conflictedArg [%conflictedSection] %j
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%conflicted) : (!fir.ref<i32>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @late_slice_free_rejection_preserves_slice(
  // CHECK-SAME: %[[SLICED:[^:]+]]: !fir.box
  // CHECK-SAME: %[[CONFLICTED:[^:]+]]: !fir.box
  // CHECK: %[[SLICED_SECTION:.*]] = fir.slice
  // CHECK: %[[CONFLICTED_SECTION:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[CONFLICTED]],
  // CHECK: fir.box_dims %[[SLICED]],
  // CHECK-NOT: fir.box_dims %[[CONFLICTED]],
  // CHECK: fir.if
  // CHECK: %[[SLICED_BYTE_BOX:.*]] = fir.convert %[[SLICED]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[SLICED_BYTES:.*]] = fir.box_addr %[[SLICED_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.coordinate_of %[[SLICED_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FAST_FLAT:.*]] = fir.array_coor %[[CONFLICTED]]
  // CHECK-NOT: fir.box_dims %[[CONFLICTED]],
  // CHECK-NOT: fir.if
  // CHECK-NOT: fir.coordinate_of
  // CHECK: %[[FAST_SLICED:.*]] = fir.array_coor %[[CONFLICTED]]
  // CHECK-SAME: [%[[CONFLICTED_SECTION]]]
  // CHECK: } else {
  // CHECK: fir.array_coor %[[SLICED]] [%[[SLICED_SECTION]]]
  // CHECK: %[[SLOW_FLAT:.*]] = fir.array_coor %[[CONFLICTED]]
  // CHECK-NOT: fir.box_dims %[[CONFLICTED]],
  // CHECK-NOT: fir.if
  // CHECK-NOT: fir.coordinate_of
  // CHECK: %[[SLOW_SLICED:.*]] = fir.array_coor %[[CONFLICTED]]
  // CHECK-SAME: [%[[CONFLICTED_SECTION]]]
  // CHECK-NOT: fir.box_dims %[[CONFLICTED]],
  // CHECK: return

  // Separate sibling owners can share the same descriptors without cloning
  // one another. Both loops return an accumulation, and the first result
  // initializes the second loop, as in sequential reduction owners.
  func.func @independent_owners_supported(
      %a: !fir.box<!fir.array<?xi32>>,
      %b: !fir.box<!fir.array<?xi32>>) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %sliceA = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %sliceB = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %first = fir.do_loop %i = %c1 to %c8 step %c1
        iter_args(%sum0 = %c0) -> (i32) {
      %x = fir.array_coor %a [%sliceA] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %b [%sliceB] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %xValue = fir.load %x : !fir.ref<i32>
      %yValue = fir.load %y : !fir.ref<i32>
      %partial0 = arith.addi %sum0, %xValue : i32
      %next0 = arith.addi %partial0, %yValue : i32
      fir.result %next0 : i32
    }
    %second = fir.do_loop %j = %c1 to %c8 step %c1
        iter_args(%sum1 = %first) -> (i32) {
      %x = fir.array_coor %a [%sliceA] %j
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %b [%sliceB] %j
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %xValue = fir.load %x : !fir.ref<i32>
      %yValue = fir.load %y : !fir.ref<i32>
      %partial1 = arith.addi %sum1, %xValue : i32
      %next1 = arith.addi %partial1, %yValue : i32
      fir.result %next1 : i32
    }
    return %second : i32
  }
  // CHECK-LABEL: func.func @independent_owners_supported(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[B:[^:]+]]: !fir.box
  // CHECK: %[[ZERO:[^ ]+]] = arith.constant 0 : i32
  // CHECK: %[[SLICE_A:[^ ]+]] = fir.slice
  // CHECK: %[[SLICE_B:[^ ]+]] = fir.slice
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.box_dims %[[B]],
  // CHECK: %[[FIRST_IF:[^ ]+]] = fir.if {{.*}} -> (i32) {
  // CHECK: %[[FIRST_A_BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[FIRST_A_BYTES:.*]] = fir.box_addr %[[FIRST_A_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[FIRST_B_BYTE_BOX:.*]] = fir.convert %[[B]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[FIRST_B_BYTES:.*]] = fir.box_addr %[[FIRST_B_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[FIRST_FAST:[^ ]+]] = fir.do_loop {{.*}} iter_args(%{{[^ ]+}} = %[[ZERO]]) -> (i32) {
  // CHECK: fir.coordinate_of %[[FIRST_A_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.coordinate_of %[[FIRST_B_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.result %{{[^ ]+}} : i32
  // CHECK: }
  // CHECK: fir.result %[[FIRST_FAST]] : i32
  // CHECK: } else {
  // CHECK: %[[FIRST_SLOW:[^ ]+]] = fir.do_loop {{.*}} iter_args(%{{[^ ]+}} = %[[ZERO]]) -> (i32) {
  // CHECK: fir.array_coor %[[A]]{{.*}}%[[SLICE_A]]
  // CHECK: fir.array_coor %[[B]]{{.*}}%[[SLICE_B]]
  // CHECK: fir.result %{{[^ ]+}} : i32
  // CHECK: }
  // CHECK: fir.result %[[FIRST_SLOW]] : i32
  // CHECK: }
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.box_dims %[[B]],
  // CHECK: %[[SECOND_IF:[^ ]+]] = fir.if {{.*}} -> (i32) {
  // CHECK: %[[SECOND_A_BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[SECOND_A_BYTES:.*]] = fir.box_addr %[[SECOND_A_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[SECOND_B_BYTE_BOX:.*]] = fir.convert %[[B]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[SECOND_B_BYTES:.*]] = fir.box_addr %[[SECOND_B_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[SECOND_FAST:[^ ]+]] = fir.do_loop {{.*}} iter_args(%{{[^ ]+}} = %[[FIRST_IF]]) -> (i32) {
  // CHECK: fir.coordinate_of %[[SECOND_A_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.coordinate_of %[[SECOND_B_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.result %{{[^ ]+}} : i32
  // CHECK: }
  // CHECK: fir.result %[[SECOND_FAST]] : i32
  // CHECK: } else {
  // CHECK: %[[SECOND_SLOW:[^ ]+]] = fir.do_loop {{.*}} iter_args(%{{[^ ]+}} = %[[FIRST_IF]]) -> (i32) {
  // CHECK: fir.array_coor %[[A]]{{.*}}%[[SLICE_A]]
  // CHECK: fir.array_coor %[[B]]{{.*}}%[[SLICE_B]]
  // CHECK: fir.result %{{[^ ]+}} : i32
  // CHECK: }
  // CHECK: fir.result %[[SECOND_SLOW]] : i32
  // CHECK: }
  // CHECK: return %[[SECOND_IF]] : i32

  // Sibling owners still require one descriptor-wide decision. An
  // unsupported owner keeps every use of that descriptor on the generic path.
  func.func @independent_owner_rejection_is_atomic(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %bad = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%bad] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
    }
    fir.do_loop %j = %c1 to %c8 step %c1 {
      %y = fir.array_coor %a [%good] %j
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @independent_owner_rejection_is_atomic(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[BAD:.*]] = fir.slice
  // CHECK: %[[GOOD:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[BAD_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[BAD]]]
  // CHECK: fir.call @use_i32(%[[BAD_ADDRESS]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[GOOD_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[GOOD]]]
  // CHECK: fir.call @use_i32(%[[GOOD_ADDRESS]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // Descriptor-wide rejection is independent of owner discovery order. A
  // supported owner encountered first must remain generic when a later owner
  // of the same descriptor has an unsupported section step.
  func.func @independent_owner_late_rejection_is_atomic(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %bad = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
    }
    fir.do_loop %j = %c1 to %c8 step %c1 {
      %y = fir.array_coor %a [%bad] %j
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @independent_owner_late_rejection_is_atomic(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[GOOD:.*]] = fir.slice
  // CHECK: %[[BAD:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop %[[I:[^ ]+]] =
  // CHECK: %[[GOOD_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[GOOD]]] %[[I]]
  // CHECK: fir.call @use_i32(%[[GOOD_ADDRESS]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop %[[J:[^ ]+]] =
  // CHECK: %[[BAD_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[BAD]]] %[[J]]
  // CHECK: fir.call @use_i32(%[[BAD_ADDRESS]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // Independent nested owners are transformed in post-order. The inner
  // versioning is therefore already present when the outer owner is cloned and
  // must survive in both versions of that outer loop.
  func.func @nested_independent_owners_supported(
      %outer: !fir.box<!fir.array<?xi32>>,
      %inner: !fir.box<!fir.array<?xf64>>, %innerLower: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %outerSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %a = fir.array_coor %outer [%outerSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%a) : (!fir.ref<i32>) -> ()
      %innerSlice = fir.slice %innerLower, %c8, %c1
        : (index, index, index) -> !fir.slice<1>
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %b = fir.array_coor %inner [%innerSlice] %j
          : (!fir.box<!fir.array<?xf64>>, !fir.slice<1>, index)
            -> !fir.ref<f64>
        fir.call @use_f64(%b) : (!fir.ref<f64>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @nested_independent_owners_supported(
  // CHECK-SAME: %[[OUTER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[INNER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[INNER_LOWER:[^:]+]]: index
  // CHECK: %[[OUTER_SLICE:.*]] = fir.slice
  // CHECK: %[[OUTER_D0:.*]]:3 = fir.box_dims %[[OUTER]],
  // CHECK: %[[OUTER_SIZE:.*]] = arith.constant 4 : index
  // CHECK: %[[OUTER_PRED:.*]] = arith.cmpi eq, %[[OUTER_D0]]#2, %[[OUTER_SIZE]]
  // CHECK: fir.if %[[OUTER_PRED]]
  // CHECK: %[[OUTER_BYTE_BOX:.*]] = fir.convert %[[OUTER]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[OUTER_BYTES:.*]] = fir.box_addr %[[OUTER_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop
  // CHECK: %[[OUTER_FAST_ADDRESS:.*]] = fir.coordinate_of %[[OUTER_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[OUTER_FAST_REF:.*]] = fir.convert %[[OUTER_FAST_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[OUTER_FAST_REF]])
  // CHECK: %[[INNER_SLICE_FAST:.*]] = fir.slice %[[INNER_LOWER]],
  // CHECK: %[[INNER_FAST_D0:.*]]:3 = fir.box_dims %[[INNER]],
  // CHECK: %[[INNER_FAST_SIZE:.*]] = arith.constant 8 : index
  // CHECK: %[[INNER_FAST_PRED:.*]] = arith.cmpi eq, %[[INNER_FAST_D0]]#2, %[[INNER_FAST_SIZE]]
  // CHECK: fir.if %[[INNER_FAST_PRED]]
  // CHECK: %[[INNER_FAST_BYTE_BOX:.*]] = fir.convert %[[INNER]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[INNER_FAST_BYTES:.*]] = fir.box_addr %[[INNER_FAST_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop
  // CHECK: %[[INNER_FAST_ADDRESS:.*]] = fir.coordinate_of %[[INNER_FAST_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[INNER_FAST_REF:.*]] = fir.convert %[[INNER_FAST_ADDRESS]]
  // CHECK: fir.call @use_f64(%[[INNER_FAST_REF]])
  // CHECK: } else {
  // CHECK: %[[INNER_FAST_FALLBACK:.*]] = fir.array_coor %[[INNER]]{{.*}}[%[[INNER_SLICE_FAST]]]
  // CHECK: fir.call @use_f64(%[[INNER_FAST_FALLBACK]])
  // CHECK: } else {
  // CHECK: fir.do_loop
  // CHECK: %[[OUTER_FALLBACK:.*]] = fir.array_coor %[[OUTER]]{{.*}}[%[[OUTER_SLICE]]]
  // CHECK: fir.call @use_i32(%[[OUTER_FALLBACK]])
  // CHECK: %[[INNER_SLICE_SLOW:.*]] = fir.slice %[[INNER_LOWER]],
  // CHECK: %[[INNER_SLOW_D0:.*]]:3 = fir.box_dims %[[INNER]],
  // CHECK: %[[INNER_SLOW_SIZE:.*]] = arith.constant 8 : index
  // CHECK: %[[INNER_SLOW_PRED:.*]] = arith.cmpi eq, %[[INNER_SLOW_D0]]#2, %[[INNER_SLOW_SIZE]]
  // CHECK: fir.if %[[INNER_SLOW_PRED]]
  // CHECK: %[[INNER_SLOW_BYTE_BOX:.*]] = fir.convert %[[INNER]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[INNER_SLOW_BYTES:.*]] = fir.box_addr %[[INNER_SLOW_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop
  // CHECK: %[[INNER_SLOW_ADDRESS:.*]] = fir.coordinate_of %[[INNER_SLOW_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[INNER_SLOW_REF:.*]] = fir.convert %[[INNER_SLOW_ADDRESS]]
  // CHECK: fir.call @use_f64(%[[INNER_SLOW_REF]])
  // CHECK: } else {
  // CHECK: %[[INNER_SLOW_FALLBACK:.*]] = fir.array_coor %[[INNER]]{{.*}}[%[[INNER_SLICE_SLOW]]]
  // CHECK: fir.call @use_f64(%[[INNER_SLOW_FALLBACK]])
  // DISABLED-LABEL: func.func @nested_independent_owners_supported(
  // DISABLED-SAME: %[[OUTER:[^:]+]]: !fir.box
  // DISABLED-SAME: %[[INNER:[^:]+]]: !fir.box
  // DISABLED-NOT: fir.if
  // DISABLED: %[[OUTER_ADDRESS:.*]] = fir.array_coor %[[OUTER]]
  // DISABLED: fir.call @use_i32(%[[OUTER_ADDRESS]])
  // DISABLED: %[[INNER_ADDRESS:.*]] = fir.array_coor %[[INNER]]
  // DISABLED: fir.call @use_f64(%[[INNER_ADDRESS]])
  // DISABLED-NOT: fir.array_coor
  // DISABLED-NOT: fir.call @use_i32
  // DISABLED-NOT: fir.call @use_f64
  // DISABLED-NOT: fir.if
  // DISABLED: return

  // Ancestor and descendant owners of the same descriptor stay generic
  // because cloning the ancestor would invalidate the descendant's plan.
  func.func @nested_owners_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %y = fir.array_coor %a [%slice] %j
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @nested_owners_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[OUTER_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[OUTER_ADDRESS]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[INNER_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[INNER_ADDRESS]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // Subtree intervals must remain transitive through a loop that has no use of
  // the descriptor. The outer and innermost owners are still an
  // ancestor/descendant pair and must therefore remain generic together.
  func.func @three_level_owners_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %outer = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%outer) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        fir.do_loop %k = %c1 to %c8 step %c1 {
          %inner = fir.array_coor %a [%slice] %k
            : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
              -> !fir.ref<i32>
          fir.call @use_i32(%inner) : (!fir.ref<i32>) -> ()
        }
      }
    }
    return
  }
  // CHECK-LABEL: func.func @three_level_owners_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[OUTER:.*]] = fir.array_coor %[[A]] [%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[OUTER]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[INNER:.*]] = fir.array_coor %[[A]] [%[[SLICE]]]
  // CHECK: fir.call @use_i32(%[[INNER]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: return

  // A flat owner nested above or below a sliced owner prevents slice
  // plans in either ordering. Unrelated slice-free versioning remains
  // permitted.
  func.func @nested_slice_flat_rejected(
      %outerSlice: !fir.box<!fir.array<?xi32>>,
      %innerSlice: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %a = fir.array_coor %outerSlice [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %b = fir.array_coor %innerSlice %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%a) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%b) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %c = fir.array_coor %outerSlice %j
          : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
        %d = fir.array_coor %innerSlice [%slice] %j
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%c) : (!fir.ref<i32>) -> ()
        fir.call @use_i32(%d) : (!fir.ref<i32>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @nested_slice_flat_rejected(
  // CHECK-SAME: %[[OUTER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[INNER:[^:]+]]: !fir.box
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[OUTER]]{{.*}}[%{{.*}}]
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[INNER]]{{.*}}[%{{.*}}]
  // CHECK-NOT: !fir.ref<!fir.array<?xi8>>
  // CHECK: return

  // A nested non-loop region remains part of the nearest enclosing loop's
  // direct-use set. The exact access is found and rewritten inside the clone.
  func.func @nested_non_loop_region_supported(
      %a: !fir.box<!fir.array<?xi32>>, %take: i1) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      fir.if %take {
        %address = fir.array_coor %a [%slice] %i
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
      }
    }
    return
  }
  // CHECK-LABEL: func.func @nested_non_loop_region_supported(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[TAKE:[^:]+]]: i1
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]],
  // CHECK: %[[PRED:.*]] = arith.cmpi eq, %[[D0]]#2,
  // CHECK: fir.if %[[PRED]]
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop
  // CHECK: fir.if %[[TAKE]]
  // CHECK: %[[FAST_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FAST:.*]] = fir.convert %[[FAST_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[FAST]])
  // CHECK: } else {
  // CHECK: fir.do_loop
  // CHECK: fir.if %[[TAKE]]
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]]
  // CHECK: fir.call @use_i32(%[[FALLBACK]])
  // CHECK: return

  // An unknown section step rejects only its descriptor. Static-unit
  // descriptors discovered before and after it retain their byte fast paths.
  func.func @dynamic_step_rejected(
      %firstArg: !fir.box<!fir.array<?xi32>>,
      %badArg: !fir.box<!fir.array<?xi32>>,
      %goodArg: !fir.box<!fir.array<?xi32>>, %dynamicStep: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %badSlice = fir.slice %c1, %c8, %dynamicStep
      : (index, index, index) -> !fir.slice<1>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %first = fir.array_coor %firstArg [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %bad = fir.array_coor %badArg [%badSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %good = fir.array_coor %goodArg [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%first) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%bad) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%good) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @dynamic_step_rejected(
  // CHECK-SAME: %[[FIRST:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[STEP:[^:]+]]: index
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.box_dims %[[FIRST]],
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK-NOT: arith.cmpi eq, {{.*}}%[[STEP]]
  // CHECK: fir.if
  // CHECK: %[[FIRST_BYTE_BOX:.*]] = fir.convert %[[FIRST]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[FIRST_BYTES:.*]] = fir.box_addr %[[FIRST_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[FIRST_ADDRESS:.*]] = fir.coordinate_of %[[FIRST_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FIRST_FAST:.*]] = fir.convert %[[FIRST_ADDRESS]]
  // CHECK: %[[BAD_FAST:.*]] = fir.array_coor %[[BAD]]
  // CHECK: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[FIRST_FAST]])
  // CHECK: fir.call @use_i32(%[[BAD_FAST]])
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK: } else {
  // CHECK: %[[FIRST_FALLBACK:.*]] = fir.array_coor %[[FIRST]]
  // CHECK: %[[BAD_FALLBACK:.*]] = fir.array_coor %[[BAD]]
  // CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD]]
  // CHECK: fir.call @use_i32(%[[FIRST_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[BAD_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[GOOD_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: return

  // This patch consumes the canonical frontend rebox(declare(pack(arg)))
  // order. The alternative declare(rebox(arg)) order is an explicit scope
  // ceiling, not a preferred long-term normalization rule.
  func.func @wrapper_order_scope_ceiling(
      %goodArg: !fir.box<!fir.array<?xi32>>,
      %boundaryArg: !fir.box<!fir.array<?xi32>>) {
    %packed = fir.pack_array %goodArg heap whole
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %declared = fir.declare %packed {uniq_name = "supported"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %reboxed = fir.rebox %declared
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %boundaryRebox = fir.rebox %boundaryArg
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %boundaryDeclared = fir.declare %boundaryRebox {uniq_name = "boundary"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %supported = fir.array_coor %reboxed [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %boundary = fir.array_coor %boundaryDeclared [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%supported) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%boundary) : (!fir.ref<i32>) -> ()
    }
    fir.unpack_array %packed to %goodArg heap
      : !fir.box<!fir.array<?xi32>>
    return
  }
  // CHECK-LABEL: func.func @wrapper_order_scope_ceiling(
  // CHECK-SAME: %[[GOOD_ARG:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BOUNDARY_ARG:[^:]+]]: !fir.box
  // CHECK: %[[PACKED:.*]] = fir.pack_array %[[GOOD_ARG]] heap whole
  // CHECK: %[[DECLARED:.*]] = fir.declare %[[PACKED]]
  // CHECK: %[[REBOXED:.*]] = fir.rebox %[[DECLARED]]
  // CHECK: %[[BOUNDARY_REBOX:.*]] = fir.rebox %[[BOUNDARY_ARG]]
  // CHECK: %[[BOUNDARY_DECLARED:.*]] = fir.declare %[[BOUNDARY_REBOX]]
  // CHECK-NOT: fir.box_dims %[[BOUNDARY_DECLARED]],
  // CHECK: fir.box_dims %[[REBOXED]],
  // CHECK-NOT: fir.box_dims %[[BOUNDARY_DECLARED]],
  // CHECK: fir.if
  // CHECK: %[[SUPPORTED_BOX:.*]] = fir.convert %[[REBOXED]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[SUPPORTED_BYTES:.*]] = fir.box_addr %[[SUPPORTED_BOX]]
  // CHECK: %[[SUPPORTED_ADDRESS:.*]] = fir.coordinate_of %[[SUPPORTED_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[SUPPORTED_FAST:.*]] = fir.convert %[[SUPPORTED_ADDRESS]]
  // CHECK: %[[BOUNDARY_FAST:.*]] = fir.array_coor %[[BOUNDARY_DECLARED]]
  // CHECK: fir.call @use_i32(%[[SUPPORTED_FAST]])
  // CHECK: fir.call @use_i32(%[[BOUNDARY_FAST]])
  // CHECK: } else {
  // CHECK: %[[SUPPORTED_FALLBACK:.*]] = fir.array_coor %[[REBOXED]]
  // CHECK: %[[BOUNDARY_FALLBACK:.*]] = fir.array_coor %[[BOUNDARY_DECLARED]]
  // CHECK: fir.call @use_i32(%[[SUPPORTED_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[BOUNDARY_FALLBACK]])
  // CHECK: fir.unpack_array %[[PACKED]] to %[[GOOD_ARG]] heap
  // CHECK-NOT: fir.box_dims %[[BOUNDARY_DECLARED]],
  // CHECK: return

  // Distinct concrete wrappers remain distinct descriptor groups even when
  // normalization traces both of them to the same function argument.
  func.func @wrapper_identity_is_preserved(
      %arg: !fir.box<!fir.array<?xi32>>) {
    %declared = fir.declare %arg {uniq_name = "declared"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %reboxed = fir.rebox %declared
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %declaredAddress = fir.array_coor %declared [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %reboxedAddress = fir.array_coor %reboxed [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%declaredAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%reboxedAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @wrapper_identity_is_preserved(
  // CHECK-SAME: %[[ARG:[^:]+]]: !fir.box
  // CHECK: %[[DECLARED:.*]] = fir.declare %[[ARG]]
  // CHECK: %[[REBOXED:.*]] = fir.rebox %[[DECLARED]]
  // CHECK: fir.box_dims %[[DECLARED]],
  // CHECK: fir.box_dims %[[REBOXED]],
  // CHECK: fir.if
  // CHECK: %[[DECLARED_BYTE_BOX:.*]] = fir.convert %[[DECLARED]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[DECLARED_BYTES:.*]] = fir.box_addr %[[DECLARED_BYTE_BOX]]
  // CHECK: %[[REBOXED_BYTE_BOX:.*]] = fir.convert %[[REBOXED]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[REBOXED_BYTES:.*]] = fir.box_addr %[[REBOXED_BYTE_BOX]]
  // CHECK: %[[DECLARED_BYTE_ADDRESS:.*]] = fir.coordinate_of
  // CHECK-SAME: %[[DECLARED_BYTES]],
  // CHECK: %[[DECLARED_FAST:.*]] = fir.convert %[[DECLARED_BYTE_ADDRESS]]
  // CHECK: %[[REBOXED_BYTE_ADDRESS:.*]] = fir.coordinate_of
  // CHECK-SAME: %[[REBOXED_BYTES]],
  // CHECK: %[[REBOXED_FAST:.*]] = fir.convert %[[REBOXED_BYTE_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[DECLARED_FAST]])
  // CHECK: fir.call @use_i32(%[[REBOXED_FAST]])
  // CHECK: } else {
  // CHECK: %[[DECLARED_FALLBACK:.*]] = fir.array_coor %[[DECLARED]]
  // CHECK: %[[REBOXED_FALLBACK:.*]] = fir.array_coor %[[REBOXED]]
  // CHECK: fir.call @use_i32(%[[DECLARED_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[REBOXED_FALLBACK]])
  // CHECK: return

  // A rebox with a non-unit section step may produce a non-contiguous view.
  // Descriptor normalization must stop at that physical descriptor.
  func.func @noncontiguous_rebox_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %noncontiguous = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    %reboxed = fir.rebox %a [%noncontiguous]
      : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>)
        -> !fir.box<!fir.array<?xi32>>
    %direct = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %reboxed [%direct] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @noncontiguous_rebox_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[NONCONTIGUOUS:.*]] = fir.slice
  // CHECK: %[[REBOXED:.*]] = fir.rebox %[[A]] [%[[NONCONTIGUOUS]]]
  // CHECK: %[[DIRECT:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[REBOXED]],
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[ADDRESS:.*]] = fir.array_coor %[[REBOXED]] {{.*}}[%[[DIRECT]]]
  // CHECK: fir.call @use_i32(%[[ADDRESS]])
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.box_dims %[[REBOXED]],
  // CHECK-NOT: fir.if
  // CHECK: return

  // Unsafe slice operands and a mismatched result type fail closed without
  // disabling an independent descriptor in the same owner.
  func.func @fail_closed_slice_forms(
      %badTriple: !fir.box<!fir.array<?xi32>>,
      %badLower: !fir.box<!fir.array<?xi32>>,
      %badResult: !fir.box<!fir.array<?xf32>>,
      %badCharacter: !fir.ref<!fir.array<?x!fir.char<1,?>>>,
      %badRecord: !fir.box<!fir.array<?x!fir.type<slice_element{i:i32}>>>,
      %good: !fir.box<!fir.array<?xi32>>, %narrowLower: ui32, %len: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %unsupported = fir.slice %c1, %c8, %undef
      : (index, index, index) -> !fir.slice<1>
    %narrowUnsigned = fir.slice %narrowLower, %c8, %c1
      : (ui32, index, index) -> !fir.slice<1>
    %unit = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %tripleAddress = fir.array_coor %badTriple [%unsupported] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %lowerAddress = fir.array_coor %badLower [%narrowUnsigned] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %resultAddress = fir.array_coor %badResult [%unit] %i
        : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %characterAddress = fir.array_coor %badCharacter [%unit] %i typeparams %len
        : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>, index,
           index) -> !fir.ref<!fir.char<1,?>>
      %recordAddress = fir.array_coor %badRecord [%unit] %i
        : (!fir.box<!fir.array<?x!fir.type<slice_element{i:i32}>>>,
           !fir.slice<1>, index) -> !fir.ref<!fir.type<slice_element{i:i32}>>
      %goodAddress = fir.array_coor %good [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%tripleAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%lowerAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%resultAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_char(%characterAddress)
        : (!fir.ref<!fir.char<1,?>>) -> ()
      fir.call @use_record(%recordAddress)
        : (!fir.ref<!fir.type<slice_element{i:i32}>>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @fail_closed_slice_forms(
  // CHECK-SAME: %[[BAD_TRIPLE:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD_LOWER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD_RESULT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[BAD_CHARACTER:[^:]+]]: !fir.ref
  // CHECK-SAME: %[[BAD_RECORD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[BAD_TRIPLE_SLICE:.*]] = fir.slice
  // CHECK: %[[NARROW_UNSIGNED:.*]] = fir.slice
  // CHECK: %[[UNIT:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[BAD_TRIPLE]],
  // CHECK-NOT: fir.box_dims %[[BAD_LOWER]],
  // CHECK-NOT: fir.box_dims %[[BAD_RESULT]],
  // CHECK-NOT: fir.box_dims %[[BAD_CHARACTER]],
  // CHECK-NOT: fir.box_dims %[[BAD_RECORD]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD_TRIPLE]],
  // CHECK-NOT: fir.box_dims %[[BAD_LOWER]],
  // CHECK-NOT: fir.box_dims %[[BAD_RESULT]],
  // CHECK-NOT: fir.box_dims %[[BAD_CHARACTER]],
  // CHECK-NOT: fir.box_dims %[[BAD_RECORD]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[BAD_TRIPLE]] [%[[BAD_TRIPLE_SLICE]]]
  // CHECK: fir.array_coor %[[BAD_LOWER]] [%[[NARROW_UNSIGNED]]]
  // CHECK: fir.array_coor %[[BAD_RESULT]] [%[[UNIT]]]
  // CHECK: fir.array_coor %[[BAD_CHARACTER]] [%[[UNIT]]]
  // CHECK: fir.array_coor %[[BAD_RECORD]] [%[[UNIT]]]
  // CHECK: fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[BAD_TRIPLE]] [%[[BAD_TRIPLE_SLICE]]]
  // CHECK: fir.array_coor %[[BAD_LOWER]] [%[[NARROW_UNSIGNED]]]
  // CHECK: fir.array_coor %[[BAD_RESULT]] [%[[UNIT]]]
  // CHECK: fir.array_coor %[[BAD_CHARACTER]] [%[[UNIT]]]
  // CHECK: fir.array_coor %[[BAD_RECORD]] [%[[UNIT]]]
  // CHECK: fir.array_coor %[[GOOD]] [%[[UNIT]]]
  // CHECK: return

  // A component path is outside the direct-sequence byte formula. Keep that
  // descriptor generic while independently versioning a supported descriptor
  // in the same owner. The FIR verifier does not tie a slice field path to the
  // memref element type, so this fixture reaches the path gate directly.
  func.func @component_path_rejected(
      %component: !fir.box<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %field = fir.field_index j, !fir.type<t{i:i32,j:i32}>
    %componentSlice = fir.slice %c1, %c8, %c1 path %field
      : (index, index, index, !fir.field) -> !fir.slice<1>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %componentAddress = fir.array_coor %component [%componentSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%componentAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @component_path_rejected(
  // CHECK-SAME: %[[COMPONENT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[COMPONENT_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[COMPONENT]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[COMPONENT]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[COMPONENT_FAST:.*]] = fir.array_coor %[[COMPONENT]] [%[[COMPONENT_SLICE]]]
  // CHECK: %[[GOOD_FAST_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_FAST_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[COMPONENT_FAST]])
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK: } else {
  // CHECK: %[[COMPONENT_FALLBACK:.*]] = fir.array_coor %[[COMPONENT]] [%[[COMPONENT_SLICE]]]
  // CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD]] [%[[GOOD_SLICE]]]
  // CHECK: fir.call @use_i32(%[[COMPONENT_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[GOOD_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[COMPONENT]],
  // CHECK: return

  // An unsupported triple in a trailing dimension must reject its descriptor
  // independently of the leading-Section rule. Otherwise the emitter could
  // misclassify it as Scalar and omit the generic-path step arithmetic.
  func.func @trailing_unsupported_slice_triple(
      %bad: !fir.box<!fir.array<?x?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %badSlice = fir.slice %c1, %c8, %c1,
                          %undef, %c8, %c4
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %badAddress = fir.array_coor %bad [%badSlice] %i, %i
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%badAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @trailing_unsupported_slice_triple(
  // CHECK-SAME: %[[BAD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[BAD_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[BAD_FAST:.*]] = fir.array_coor %[[BAD]] [%[[BAD_SLICE]]]
  // CHECK: %[[GOOD_FAST_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_FAST_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[BAD_FAST]])
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK: } else {
  // CHECK: %[[BAD_FALLBACK:.*]] = fir.array_coor %[[BAD]] [%[[BAD_SLICE]]]
  // CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD]] [%[[GOOD_SLICE]]]
  // CHECK: fir.call @use_i32(%[[BAD_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[GOOD_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: return

  // A sliced access with fewer coordinates than the descriptor rank remains
  // generic without disabling an independent full-rank access in the owner.
  func.func @reduced_rank_rejected(
      %bad: !fir.box<!fir.array<?x?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %badSlice = fir.slice %c1, %c8, %c1,
                          %c1, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %goodSlice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %badAddress = fir.array_coor %bad [%badSlice] %i
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%goodSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%badAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @reduced_rank_rejected(
  // CHECK-SAME: %[[BAD:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[BAD_SLICE:.*]] = fir.slice
  // CHECK: %[[GOOD_SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[FAST_I:[^ ]+]] =
  // CHECK: %[[BAD_FAST:.*]] = fir.array_coor %[[BAD]] [%[[BAD_SLICE]]] %[[FAST_I]]
  // CHECK-NOT: fir.array_coor %[[GOOD]]
  // CHECK: %[[GOOD_BYTE_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_BYTE_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[BAD_FAST]])
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK: } else {
  // CHECK: fir.do_loop %[[SLOW_I:[^ ]+]] =
  // CHECK: %[[BAD_FALLBACK:.*]] = fir.array_coor %[[BAD]] [%[[BAD_SLICE]]] %[[SLOW_I]]
  // CHECK: %[[GOOD_FALLBACK:.*]] = fir.array_coor %[[GOOD]] [%[[GOOD_SLICE]]] %[[SLOW_I]]
  // CHECK: fir.call @use_i32(%[[BAD_FALLBACK]])
  // CHECK: fir.call @use_i32(%[[GOOD_FALLBACK]])
  // CHECK-NOT: fir.box_dims %[[BAD]],
  // CHECK: return

  // A non-defining slice block argument cannot be decoded and remains generic.
  func.func @opaque_slice_rejected(
      %a: !fir.box<!fir.array<?xi32>>, %slice: !fir.slice<1>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @opaque_slice_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[SLICE:[^:]+]]: !fir.slice<1>
  // CHECK-NOT: fir.box_dims
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: fir.array_coor %[[A]] {{.*}}[%[[SLICE]]]
  // CHECK-NOT: fir.if
  // CHECK: return

  // A descriptor wrapper created inside the loop does not dominate the loop
  // owner. Its slice must remain generic rather than moving descriptor
  // metadata reads above the wrapper definition.
  func.func @loop_local_descriptor_rejected(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %local = fir.rebox %a
        : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
      %address = fir.array_coor %local [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @loop_local_descriptor_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-NOT: fir.box_dims
  // CHECK-NOT: fir.if
  // CHECK: fir.do_loop
  // CHECK: %[[LOCAL:.*]] = fir.rebox %[[A]]
  // CHECK: %[[ADDRESS:.*]] = fir.array_coor %[[LOCAL]]{{.*}}[%{{.*}}]
  // CHECK: fir.call @use_i32(%[[ADDRESS]])
  // CHECK-NOT: fir.if
  // CHECK: return

  // Leading-Scalar slices and non-shape lower-bound carriers are future
  // extensions. These rejected forms do not affect an independent descriptor.
  func.func @unsupported_shapes_rejected(
      %allScalar: !fir.box<!fir.array<?x?xi32>>,
      %leadingScalar: !fir.box<!fir.array<?x?xi32>>,
      %shifted: !fir.box<!fir.array<?xi32>>,
      %shapeShifted: !fir.box<!fir.array<?xi32>>,
      %opaque: !fir.box<!fir.array<?xi32>>, %opaqueShape: !fir.shape<1>,
      %good: !fir.box<!fir.array<?xi32>>, %origin: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %scalarSlice = fir.slice %c1, %undef, %undef,
                             %c1, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %leadingScalarSlice = fir.slice %c1, %undef, %undef,
                                    %c1, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %unit = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %shift = fir.shift %origin : (index) -> !fir.shift<1>
    %shapeShift = fir.shape_shift %origin, %c8
      : (index, index) -> !fir.shapeshift<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %allScalar [%scalarSlice] %c1, %c1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      %v = fir.array_coor %leadingScalar [%leadingScalarSlice] %c1, %i
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %shifted(%shift) [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %w = fir.array_coor %shapeShifted(%shapeShift) [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>,
           index) -> !fir.ref<i32>
      %o = fir.array_coor %opaque(%opaqueShape) [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %z = fir.array_coor %good [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%v) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%w) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%o) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%z) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @unsupported_shapes_rejected(
  // CHECK-SAME: %[[SCALAR:[^:]+]]: !fir.box
  // CHECK-SAME: %[[LEADING_SCALAR:[^:]+]]: !fir.box
  // CHECK-SAME: %[[SHIFTED:[^:]+]]: !fir.box
  // CHECK-SAME: %[[SHAPE_SHIFTED:[^:]+]]: !fir.box
  // CHECK-SAME: %[[OPAQUE:[^:]+]]: !fir.box
  // CHECK-SAME: %[[OPAQUE_SHAPE:[^:]+]]: !fir.shape<1>
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[SCALAR_SLICE:.*]] = fir.slice
  // CHECK: %[[LEADING_SLICE:.*]] = fir.slice
  // CHECK: %[[UNIT:.*]] = fir.slice
  // CHECK: %[[SHIFT:.*]] = fir.shift
  // CHECK: %[[SHAPE_SHIFT:.*]] = fir.shape_shift
  // CHECK-NOT: fir.box_dims %[[SCALAR]],
  // CHECK-NOT: fir.box_dims %[[LEADING_SCALAR]],
  // CHECK-NOT: fir.box_dims %[[SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[SHAPE_SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[OPAQUE]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[SCALAR]],
  // CHECK-NOT: fir.box_dims %[[LEADING_SCALAR]],
  // CHECK-NOT: fir.box_dims %[[SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[SHAPE_SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[OPAQUE]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[SCALAR]] [%[[SCALAR_SLICE]]]
  // CHECK: fir.array_coor %[[LEADING_SCALAR]] [%[[LEADING_SLICE]]]
  // CHECK: fir.array_coor %[[SHIFTED]](%[[SHIFT]]) [%[[UNIT]]]
  // CHECK: fir.array_coor %[[SHAPE_SHIFTED]](%[[SHAPE_SHIFT]]) [%[[UNIT]]]
  // CHECK: fir.array_coor %[[OPAQUE]](%[[OPAQUE_SHAPE]]) [%[[UNIT]]]
  // CHECK: fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: } else {
  // CHECK: fir.array_coor %[[SCALAR]] [%[[SCALAR_SLICE]]]
  // CHECK: fir.array_coor %[[LEADING_SCALAR]] [%[[LEADING_SLICE]]]
  // CHECK: fir.array_coor %[[SHIFTED]](%[[SHIFT]]) [%[[UNIT]]]
  // CHECK: fir.array_coor %[[SHAPE_SHIFTED]](%[[SHAPE_SHIFT]]) [%[[UNIT]]]
  // CHECK: fir.array_coor %[[OPAQUE]](%[[OPAQUE_SHAPE]]) [%[[UNIT]]]
  // CHECK: fir.array_coor %[[GOOD]] [%[[UNIT]]]
  // CHECK-NOT: fir.box_dims %[[SCALAR]],
  // CHECK-NOT: fir.box_dims %[[LEADING_SCALAR]],
  // CHECK-NOT: fir.box_dims %[[SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[SHAPE_SHIFTED]],
  // CHECK-NOT: fir.box_dims %[[OPAQUE]],
  // CHECK: return

  // CFI_MAX_RANK is the inclusive slice limit. The next rank must stay
  // generic without indexing the fixed-size dimension metadata array.
  func.func @rank15_accepted(
      %a: !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xi32>>,
      %middle: index, %last: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape %c8, %c8, %c8, %c8, %c8, %c8, %c8, %c8,
                       %c8, %c8, %c8, %c8, %c8, %c8, %c8
      : (index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index) -> !fir.shape<15>
    %slice = fir.slice %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1
      : (index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index)
        -> !fir.slice<15>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice]
        %i, %i, %i, %i, %i, %i, %i, %middle,
        %i, %i, %i, %i, %i, %i, %last
        : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xi32>>,
           !fir.shape<15>, !fir.slice<15>, index, index, index, index,
           index, index, index, index, index, index, index, index, index,
           index, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @rank15_accepted(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[MIDDLE:[^:]+]]: index
  // CHECK-SAME: %[[LAST:[^:]+]]: index
  // CHECK: %[[SHAPE:.*]] = fir.shape
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: %[[DIM0:.*]] = arith.constant 0 : index
  // CHECK: %[[D0:.*]]:3 = fir.box_dims %[[A]], %[[DIM0]]
  // CHECK: %[[ELEMENT_SIZE:.*]] = arith.constant 4 : index
  // CHECK: arith.cmpi eq, %[[D0]]#2, %[[ELEMENT_SIZE]] : index
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[D1:.*]]:3 = fir.box_dims %[[A]], %[[DIM1]]
  // CHECK: %[[DIM2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[D2:.*]]:3 = fir.box_dims %[[A]], %[[DIM2]]
  // CHECK: %[[DIM3:.*]] = arith.constant 3 : index
  // CHECK-NEXT: %[[D3:.*]]:3 = fir.box_dims %[[A]], %[[DIM3]]
  // CHECK: %[[DIM4:.*]] = arith.constant 4 : index
  // CHECK-NEXT: %[[D4:.*]]:3 = fir.box_dims %[[A]], %[[DIM4]]
  // CHECK: %[[DIM5:.*]] = arith.constant 5 : index
  // CHECK-NEXT: %[[D5:.*]]:3 = fir.box_dims %[[A]], %[[DIM5]]
  // CHECK: %[[DIM6:.*]] = arith.constant 6 : index
  // CHECK-NEXT: %[[D6:.*]]:3 = fir.box_dims %[[A]], %[[DIM6]]
  // CHECK: %[[DIM7:.*]] = arith.constant 7 : index
  // CHECK-NEXT: %[[D7:.*]]:3 = fir.box_dims %[[A]], %[[DIM7]]
  // CHECK: %[[DIM8:.*]] = arith.constant 8 : index
  // CHECK-NEXT: %[[D8:.*]]:3 = fir.box_dims %[[A]], %[[DIM8]]
  // CHECK: %[[DIM9:.*]] = arith.constant 9 : index
  // CHECK-NEXT: %[[D9:.*]]:3 = fir.box_dims %[[A]], %[[DIM9]]
  // CHECK: %[[DIM10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[D10:.*]]:3 = fir.box_dims %[[A]], %[[DIM10]]
  // CHECK: %[[DIM11:.*]] = arith.constant 11 : index
  // CHECK-NEXT: %[[D11:.*]]:3 = fir.box_dims %[[A]], %[[DIM11]]
  // CHECK: %[[DIM12:.*]] = arith.constant 12 : index
  // CHECK-NEXT: %[[D12:.*]]:3 = fir.box_dims %[[A]], %[[DIM12]]
  // CHECK: %[[DIM13:.*]] = arith.constant 13 : index
  // CHECK-NEXT: %[[D13:.*]]:3 = fir.box_dims %[[A]], %[[DIM13]]
  // CHECK: %[[DIM14:.*]] = arith.constant 14 : index
  // CHECK-NEXT: %[[D14:.*]]:3 = fir.box_dims %[[A]], %[[DIM14]]
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.do_loop %[[FAST_IV:[^ ]+]] =
  // CHECK: %[[OFFSET0:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES0:.*]] = arith.muli %[[OFFSET0]], %[[ELEMENT_SIZE]] : index
  // CHECK: %[[OFFSET1:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES1:.*]] = arith.muli %[[D1]]#2, %[[OFFSET1]] : index
  // CHECK: %[[SUM1:.*]] = arith.addi %[[BYTES0]], %[[BYTES1]] : index
  // CHECK: %[[OFFSET2:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES2:.*]] = arith.muli %[[D2]]#2, %[[OFFSET2]] : index
  // CHECK: %[[SUM2:.*]] = arith.addi %[[SUM1]], %[[BYTES2]] : index
  // CHECK: %[[OFFSET3:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES3:.*]] = arith.muli %[[D3]]#2, %[[OFFSET3]] : index
  // CHECK: %[[SUM3:.*]] = arith.addi %[[SUM2]], %[[BYTES3]] : index
  // CHECK: %[[OFFSET4:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES4:.*]] = arith.muli %[[D4]]#2, %[[OFFSET4]] : index
  // CHECK: %[[SUM4:.*]] = arith.addi %[[SUM3]], %[[BYTES4]] : index
  // CHECK: %[[OFFSET5:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES5:.*]] = arith.muli %[[D5]]#2, %[[OFFSET5]] : index
  // CHECK: %[[SUM5:.*]] = arith.addi %[[SUM4]], %[[BYTES5]] : index
  // CHECK: %[[OFFSET6:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES6:.*]] = arith.muli %[[D6]]#2, %[[OFFSET6]] : index
  // CHECK: %[[SUM6:.*]] = arith.addi %[[SUM5]], %[[BYTES6]] : index
  // CHECK: %[[MIDDLE_OFFSET:.*]] = arith.subi %[[MIDDLE]], %[[FAST_ONE]] : index
  // CHECK: %[[MIDDLE_BYTES:.*]] = arith.muli %[[D7]]#2, %[[MIDDLE_OFFSET]] : index
  // CHECK: %[[SUM7:.*]] = arith.addi %[[SUM6]], %[[MIDDLE_BYTES]] : index
  // CHECK: %[[OFFSET8:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES8:.*]] = arith.muli %[[D8]]#2, %[[OFFSET8]] : index
  // CHECK: %[[SUM8:.*]] = arith.addi %[[SUM7]], %[[BYTES8]] : index
  // CHECK: %[[OFFSET9:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES9:.*]] = arith.muli %[[D9]]#2, %[[OFFSET9]] : index
  // CHECK: %[[SUM9:.*]] = arith.addi %[[SUM8]], %[[BYTES9]] : index
  // CHECK: %[[OFFSET10:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES10:.*]] = arith.muli %[[D10]]#2, %[[OFFSET10]] : index
  // CHECK: %[[SUM10:.*]] = arith.addi %[[SUM9]], %[[BYTES10]] : index
  // CHECK: %[[OFFSET11:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES11:.*]] = arith.muli %[[D11]]#2, %[[OFFSET11]] : index
  // CHECK: %[[SUM11:.*]] = arith.addi %[[SUM10]], %[[BYTES11]] : index
  // CHECK: %[[OFFSET12:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES12:.*]] = arith.muli %[[D12]]#2, %[[OFFSET12]] : index
  // CHECK: %[[SUM12:.*]] = arith.addi %[[SUM11]], %[[BYTES12]] : index
  // CHECK: %[[OFFSET13:.*]] = arith.subi %[[FAST_IV]], %[[FAST_ONE]] : index
  // CHECK: %[[BYTES13:.*]] = arith.muli %[[D13]]#2, %[[OFFSET13]] : index
  // CHECK: %[[SUM13:.*]] = arith.addi %[[SUM12]], %[[BYTES13]] : index
  // CHECK: %[[LAST_OFFSET:.*]] = arith.subi %[[LAST]], %[[FAST_ONE]] : index
  // CHECK: %[[LAST_BYTES:.*]] = arith.muli %[[D14]]#2, %[[LAST_OFFSET]] : index
  // CHECK: %[[TOTAL:.*]] = arith.addi %[[SUM13]], %[[LAST_BYTES]] : index
  // CHECK: %[[BYTE_ADDRESS:.*]] = fir.coordinate_of %[[BYTES]], %[[TOTAL]]
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FAST_REF:.*]] = fir.convert %[[BYTE_ADDRESS]]
  // CHECK-NEXT: fir.call @use_i32(%[[FAST_REF]])
  // CHECK-NOT: fir.array_coor %[[A]]
  // CHECK: } else {
  // CHECK: fir.do_loop %[[SLOW_IV:[^ ]+]] =
  // CHECK: %[[FALLBACK:.*]] = fir.array_coor %[[A]](%[[SHAPE]]) [%[[SLICE]]] %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[MIDDLE]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[SLOW_IV]], %[[LAST]]
  // CHECK-NEXT: fir.call @use_i32(%[[FALLBACK]])
  // CHECK-NOT: fir.array_coor %[[A]]
  // CHECK: return

  // Slice operands defined inside the selected owner must be read from the
  // cloned loop. Using the original load in the fast branch would violate SSA
  // dominance after the original loop is moved into the fallback region.
  func.func @loop_local_slice_operand(
      %a: !fir.box<!fir.array<?xi32>>, %lowerRef: !fir.ref<i64>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %lower = fir.load %lowerRef : !fir.ref<i64>
      %slice = fir.slice %lower, %c8, %c1
        : (i64, index, index) -> !fir.slice<1>
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @loop_local_slice_operand(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK-SAME: %[[LOWER_REF:[^:]+]]: !fir.ref<i64>
  // CHECK: fir.box_dims %[[A]],
  // CHECK: fir.if
  // CHECK: %[[FAST_ONE:.*]] = arith.constant 1 : index
  // CHECK: fir.do_loop %[[FAST_I:[^ ]+]] =
  // CHECK: %[[FAST_LOWER:.*]] = fir.load %[[LOWER_REF]]
  // CHECK-NOT: fir.slice
  // CHECK: %[[FAST_LOWER_INDEX:.*]] = fir.convert %[[FAST_LOWER]] : (i64) -> index
  // CHECK: %[[FAST_ADJUST:.*]] = arith.subi %[[FAST_LOWER_INDEX]], %[[FAST_ONE]] : index
  // CHECK: arith.addi {{.*}}, %[[FAST_ADJUST]] : index
  // CHECK: %[[FAST_ADDRESS:.*]] = fir.coordinate_of
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[FAST_REF:.*]] = fir.convert %[[FAST_ADDRESS]]
  // CHECK: fir.call @use_i32(%[[FAST_REF]])
  // CHECK: } else {
  // CHECK: fir.do_loop %[[SLOW_I:[^ ]+]] =
  // CHECK: %[[SLOW_LOWER:.*]] = fir.load %[[LOWER_REF]]
  // CHECK: %[[SLOW_SLICE:.*]] = fir.slice %[[SLOW_LOWER]],
  // CHECK: %[[SLOW_ADDRESS:.*]] = fir.array_coor %[[A]] [%[[SLOW_SLICE]]] %[[SLOW_I]]
  // CHECK: fir.call @use_i32(%[[SLOW_ADDRESS]])
  // CHECK: return

  // Rank 16 exceeds CFI_MAX_RANK and must remain generic without indexing the
  // fixed-size dimension metadata array used by the versioned path.
  func.func @rank16_rejected(
      %a: !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape %c8, %c8, %c8, %c8, %c8, %c8, %c8, %c8,
                       %c8, %c8, %c8, %c8, %c8, %c8, %c8, %c8
      : (index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index)
        -> !fir.shape<16>
    %slice = fir.slice %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1,
                       %c1, %c8, %c1, %c1, %c8, %c1
      : (index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index, index, index, index, index, index, index,
         index, index, index) -> !fir.slice<16>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice]
        %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i
        : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xi32>>,
           !fir.shape<16>, !fir.slice<16>, index, index, index, index,
           index, index, index, index, index, index, index, index, index,
           index, index, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @rank16_rejected(
  // CHECK-SAME: %[[A:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[A]],
  // CHECK-NOT: fir.if
  // CHECK: fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // CHECK: return

  // Optional and volatile descriptors keep the generic path because direct
  // byte addressing must not bypass their established semantics. Verify the
  // same rejection through fir.declare and fir.pack_array wrappers. Storage
  // wrappers are outside this initial direct-sequence contract. A volatile
  // result type is rejected independently of descriptor attributes. An
  // independent direct sequence descriptor remains eligible.
  func.func @descriptor_semantics_rejected(
      %optionalRoot: !fir.box<!fir.array<?xi32>>,
      %volatile: !fir.box<!fir.array<?xi32>> {fir.volatile},
      %volatileResult: !fir.box<!fir.array<?xi32>>,
      %heap: !fir.box<!fir.heap<!fir.array<?xi32>>>,
      %pointer: !fir.box<!fir.ptr<!fir.array<?xi32>>>,
      %declaredRoot: !fir.box<!fir.array<?xi32>>,
      %packedRoot: !fir.box<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %optional = fir.rebox %optionalRoot optional
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %declaredVolatile = fir.declare %declaredRoot
        {fortran_attrs = #fir.var_attrs<volatile>,
         uniq_name = "declaredVolatile"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %declaredOptional = fir.declare %packedRoot
        {fortran_attrs = #fir.var_attrs<optional>,
         uniq_name = "declaredOptional"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %packedOptional = fir.pack_array %declaredOptional heap whole
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %packedVolatile = fir.pack_array %declaredVolatile heap whole
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %optional [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %y = fir.array_coor %volatile [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %r = fir.array_coor %volatileResult [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32, volatile>
      %h = fir.array_coor %heap [%slice] %i
        : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %p = fir.array_coor %pointer [%slice] %i
        : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %d = fir.array_coor %declaredVolatile [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %q = fir.array_coor %packedOptional [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %v = fir.array_coor %packedVolatile [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %z = fir.array_coor %good [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%y) : (!fir.ref<i32>) -> ()
      fir.call @use_volatile(%r) : (!fir.ref<i32, volatile>) -> ()
      fir.call @use_i32(%h) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%p) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%d) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%q) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%v) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%z) : (!fir.ref<i32>) -> ()
    }
    fir.unpack_array %packedOptional to %declaredOptional heap
      : !fir.box<!fir.array<?xi32>>
    fir.unpack_array %packedVolatile to %declaredVolatile heap
      : !fir.box<!fir.array<?xi32>>
    return
  }
  // CHECK-LABEL: func.func @descriptor_semantics_rejected(
  // CHECK-SAME: %[[OPTIONAL_ROOT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[VOLATILE:[^:]+]]: !fir.box
  // CHECK-SAME: %[[VOLATILE_RESULT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[HEAP:[^:]+]]: !fir.box
  // CHECK-SAME: %[[POINTER:[^:]+]]: !fir.box
  // CHECK-SAME: %[[DECLARED_ROOT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[PACKED_ROOT:[^:]+]]: !fir.box
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[OPTIONAL:.*]] = fir.rebox %[[OPTIONAL_ROOT]] optional
  // CHECK: %[[DECLARED:.*]] = fir.declare %[[DECLARED_ROOT]]
  // CHECK-SAME: fortran_attrs = #fir.var_attrs<volatile>
  // CHECK: %[[DECLARED_OPTIONAL:.*]] = fir.declare %[[PACKED_ROOT]]
  // CHECK-SAME: fortran_attrs = #fir.var_attrs<optional>
  // CHECK: %[[PACKED:.*]] = fir.pack_array %[[DECLARED_OPTIONAL]] heap whole
  // CHECK: %[[PACKED_VOLATILE:.*]] = fir.pack_array %[[DECLARED]] heap whole
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[OPTIONAL]],
  // CHECK-NOT: fir.box_dims %[[VOLATILE]],
  // CHECK-NOT: fir.box_dims %[[VOLATILE_RESULT]],
  // CHECK-NOT: fir.box_dims %[[HEAP]],
  // CHECK-NOT: fir.box_dims %[[POINTER]],
  // CHECK-NOT: fir.box_dims %[[DECLARED]],
  // CHECK-NOT: fir.box_dims %[[PACKED]],
  // CHECK-NOT: fir.box_dims %[[PACKED_VOLATILE]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[OPTIONAL]],
  // CHECK-NOT: fir.box_dims %[[VOLATILE]],
  // CHECK-NOT: fir.box_dims %[[VOLATILE_RESULT]],
  // CHECK-NOT: fir.box_dims %[[HEAP]],
  // CHECK-NOT: fir.box_dims %[[POINTER]],
  // CHECK-NOT: fir.box_dims %[[DECLARED]],
  // CHECK-NOT: fir.box_dims %[[PACKED]],
  // CHECK-NOT: fir.box_dims %[[PACKED_VOLATILE]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.array_coor %[[OPTIONAL]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[VOLATILE]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[VOLATILE_RESULT]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[HEAP]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[POINTER]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[DECLARED]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[PACKED]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[PACKED_VOLATILE]]{{.*}}[%[[SLICE]]]
  // CHECK: %[[GOOD_ADDRESS:.*]] = fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: %[[GOOD_FAST:.*]] = fir.convert %[[GOOD_ADDRESS]]
  // CHECK-NOT: fir.array_coor %[[GOOD]]
  // CHECK: fir.call @use_i32(%[[GOOD_FAST]])
  // CHECK-NOT: fir.array_coor %[[GOOD]]
  // CHECK: } else {
  // CHECK: fir.array_coor %[[OPTIONAL]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[VOLATILE]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[VOLATILE_RESULT]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[HEAP]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[POINTER]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[DECLARED]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[PACKED]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[PACKED_VOLATILE]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.array_coor %[[GOOD]]{{.*}}[%[[SLICE]]]
  // CHECK: fir.unpack_array %[[PACKED]] to %[[DECLARED_OPTIONAL]] heap
  // CHECK: fir.unpack_array %[[PACKED_VOLATILE]] to %[[DECLARED]] heap
  // CHECK: return

  // A raw assumed-shape reference is not a descriptor and must remain on the
  // generic path without disabling an independent supported descriptor.
  func.func @non_box_descriptor_rejected(
      %raw: !fir.ref<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %rawAddress = fir.array_coor %raw [%slice] %i
        : (!fir.ref<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %good [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%rawAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @non_box_descriptor_rejected(
  // CHECK-SAME: %[[RAW:[^:]+]]: !fir.ref<!fir.array<?xi32>>
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: %[[RAW_FAST:.*]] = fir.array_coor %[[RAW]] [%[[SLICE]]]
  // CHECK: fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.call @use_i32(%[[RAW_FAST]])
  // CHECK: } else {
  // CHECK: fir.array_coor %[[RAW]] [%[[SLICE]]]
  // CHECK: fir.array_coor %[[GOOD]] [%[[SLICE]]]
  // CHECK: return

  // A volatile descriptor type is rejected independently of a supported
  // non-volatile polymorphic descriptor in the same owner.
  func.func @class_descriptor_semantics(
      %good: !fir.class<!fir.array<?xi32>>,
      %volatile: !fir.class<!fir.array<?xi32>, volatile>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %goodAddress = fir.array_coor %good [%slice] %i
        : (!fir.class<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %volatileAddress = fir.array_coor %volatile [%slice] %i
        : (!fir.class<!fir.array<?xi32>, volatile>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%volatileAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }
  // CHECK-LABEL: func.func @class_descriptor_semantics(
  // CHECK-SAME: %[[GOOD:[^:]+]]: !fir.class<!fir.array<?xi32>>
  // CHECK-SAME: %[[VOLATILE:[^:]+]]: !fir.class<!fir.array<?xi32>, volatile>
  // CHECK: %[[SLICE:.*]] = fir.slice
  // CHECK-NOT: fir.box_dims %[[VOLATILE]],
  // CHECK: fir.box_dims %[[GOOD]],
  // CHECK-NOT: fir.box_dims %[[VOLATILE]],
  // CHECK: fir.if
  // CHECK: %[[GOOD_BYTE_BOX:.*]] = fir.convert %[[GOOD]]
  // CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
  // CHECK: %[[GOOD_BYTES:.*]] = fir.box_addr %[[GOOD_BYTE_BOX]]
  // CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
  // CHECK: fir.coordinate_of %[[GOOD_BYTES]],
  // CHECK-SAME: -> !fir.ref<i8>
  // CHECK: fir.array_coor %[[VOLATILE]] [%[[SLICE]]]
  // CHECK: } else {
  // CHECK: fir.array_coor %[[GOOD]] [%[[SLICE]]]
  // CHECK: fir.array_coor %[[VOLATILE]] [%[[SLICE]]]
  // CHECK: return

}
