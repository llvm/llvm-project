// RUN: fir-opt --loop-versioning %s -o %t
// RUN: FileCheck %s --input-file=%t --enable-var-scope
// RUN: FileCheck %s --input-file=%t --check-prefix=GUARDS
// RUN: fir-opt --canonicalize --loop-versioning %s -o %t.canonicalized
// RUN: FileCheck %s --input-file=%t.canonicalized \
// RUN:   --check-prefix=CANONICALIZED --enable-var-scope
// RUN: FileCheck %s --input-file=%t.canonicalized --check-prefix=CANON-GUARDS
// RUN: fir-opt --loop-versioning --canonicalize %s -o %t.postcanonicalized
// RUN: FileCheck %s --input-file=%t.postcanonicalized \
// RUN:   --check-prefix=POST-CANON --enable-var-scope
// RUN: fir-opt '--loop-versioning=enable-slices=false' %s | \
// RUN:   FileCheck %s --check-prefix=DISABLED --enable-var-scope

// Exercise accepted slice address forms, loop-local rejection, wrapper and
// type boundaries, mixed accesses, and clone-local operands. The checks focus
// on the common loop-versioning emitter: the fast clone uses flattened typed
// accesses while the fallback retains the original sliced accesses.

module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  fir.defaultkind = "a1c4d8i4l4r4",
  fir.kindmap = "i2:1,i3:0,i16:64",
  llvm.data_layout = "e-p:64:64:64:64"
} {
  func.func private @use_i32(!fir.ref<i32>)
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

  // Each access derives its flattened index from its own slice. The first is
  // [Section, Scalar], while the second is [Section, Section].
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

  // Both physical accesses intentionally share one fir.slice and are rewritten
  // by the common emitter with their own coordinates.
  func.func @two_accesses_shared_slice(
      %a: !fir.box<!fir.array<?xi32>>, %other: index) {
    %c1_i16 = arith.constant 1 : i16
    %c1_i64 = fir.convert %c1_i16 : (i16) -> i64
    %c1 = fir.convert %c1_i64 : (i64) -> index
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

  // An unsupported access rejects the descriptor within this loop.
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

  // Flat and sliced accesses to the same descriptor share one versioned loop.
  func.func @mixed_flat_and_slice_supported(
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

  // Separate sliced and flat owners are independently versioned.
  func.func @sibling_flat_and_slice_owners_supported(
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

  // The same declared descriptor may be versioned independently in sliced and
  // flat owners.
  func.func @wrapped_flat_and_slice_owners_supported(
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

  // Sequential owners and independent flat work are handled by the common
  // loop-local analysis.
  func.func @independent_mixed_owners_supported(
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

  // A raw one-bit step is rejected conservatively, while an independent
  // ordinary unit step remains eligible.
  func.func @i1_steps_rejected(
      %badStepArg: !fir.box<!fir.array<?xi32>>,
      %goodArg: !fir.box<!fir.array<?xi32>>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %badStep = fir.slice %c1, %c8, %true
      : (index, index, i1) -> !fir.slice<1>
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %badStepAddress = fir.array_coor %badStepArg [%badStep] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %goodAddress = fir.array_coor %goodArg [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%badStepAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

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

  // Unit section lowers remain legal regardless of source integer width. The
  // fast path converts each lower to index before forming its adjustment.
  func.func @static_one_lowers_allow_source_widths(
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

  // Do not treat a value that becomes one only through truncation as a unit
  // step. Its source constant is not one, so the descriptor remains generic.
  func.func @truncated_step_rejected(
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

  // A literal one remains a unit step through wider integer conversions.
  func.func @literal_one_convert_chain_step(
      %a: !fir.box<!fir.array<?xi32>>) {
    %source = arith.constant 1 : i16
    %middle = fir.convert %source : (i16) -> i8
    %step = fir.convert %middle : (i8) -> index
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

  // One owner may combine sliced and flat descriptors; both use the common
  // flattened fast path.
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


  // Nested uses are owned by their nearest loop. A mixed outer owner and an
  // independent sliced descriptor remain versionable.
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

  // An unsupported sibling owner does not block a supported owner of the same
  // descriptor.
  func.func @independent_owner_rejection_is_local(
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

  // Loop-local rejection is independent of discovery order: a later
  // unsupported owner does not retract an earlier supported owner.
  func.func @independent_owner_late_rejection_is_local(
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

  // The common clone walk rewrites uses in both ancestor and descendant loops.
  func.func @nested_owners_supported(
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

  // Nested rewriting remains valid through an intervening loop that has no
  // access to the descriptor.
  func.func @three_level_owners_supported(
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

  // A rejected access in a nested loop invalidates an earlier accepted access
  // to the same descriptor in every enclosing loop, including across an
  // intermediate loop that does not access the descriptor.
  func.func @three_level_late_rejection_blocks_ancestor(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %good = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %bad = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %outer = fir.array_coor %a [%good] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%outer) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        fir.do_loop %k = %c1 to %c8 step %c1 {
          %inner = fir.array_coor %a [%bad] %k
            : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
              -> !fir.ref<i32>
          fir.call @use_i32(%inner) : (!fir.ref<i32>) -> ()
        }
      }
    }
    return
  }

  // Nested sliced and flat accesses are both handled by the common emitter.
  func.func @nested_slice_flat_supported(
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

  // An unknown section step rejects only its descriptor. Static-unit
  // descriptors discovered before and after it retain their fast paths.
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

  // Normalization follows interleaved continuity-preserving wrappers back to
  // their function arguments, so both wrapper orders can be versioned.
  func.func @interleaved_wrapper_order(
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

  // Unsafe slice operands fail closed without disabling an independent
  // descriptor in the same owner.
  func.func @fail_closed_slice_forms(
      %badTriple: !fir.box<!fir.array<?xi32>>,
      %badCharacter: !fir.ref<!fir.array<?x!fir.char<1,?>>>,
      %badRecord: !fir.box<!fir.array<?x!fir.type<slice_element{i:i32}>>>,
      %good: !fir.box<!fir.array<?xi32>>, %len: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %unsupported = fir.slice %c1, %c8, %undef
      : (index, index, index) -> !fir.slice<1>
    %unit = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %tripleAddress = fir.array_coor %badTriple [%unsupported] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
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
      fir.call @use_char(%characterAddress)
        : (!fir.ref<!fir.char<1,?>>) -> ()
      fir.call @use_record(%recordAddress)
        : (!fir.ref<!fir.type<slice_element{i:i32}>>) -> ()
      fir.call @use_i32(%goodAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

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

  // An unsupported triple in a trailing dimension must reject its descriptor
  // independently of the leading-Section rule. Otherwise the emitter could
  // misclassify it as Scalar and omit the generic-path step arithmetic.
  func.func @trailing_unsupported_slice_triple(
      %bad: !fir.box<!fir.array<?x?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %badSlice = fir.slice %c1, %c8, %c1,
                          %undef, %c8, %c1
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

  // Scalar dimensions already carry one source coordinate each, so all-scalar
  // and leading-scalar slices use the same flat-index formula as other slices.
  func.func @scalar_slices_supported(
      %allScalar: !fir.box<!fir.array<?x?xi32>>,
      %leadingScalar: !fir.box<!fir.array<?x?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>, %origin0: index, %origin1: index,
      %scalarLower: index, %sectionLower: index, %index0: index,
      %index1: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %shape = fir.shift %origin0, %origin1 : (index, index) -> !fir.shift<2>
    %scalarSlice = fir.slice %c1, %undef, %undef,
                             %c1, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %leadingScalarSlice = fir.slice %scalarLower, %undef, %undef,
                                    %sectionLower, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    %unit = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %x = fir.array_coor %allScalar [%scalarSlice] %c1, %c1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>, index, index)
          -> !fir.ref<i32>
      %v = fir.array_coor %leadingScalar(%shape) [%leadingScalarSlice]
        %index0, %index1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.slice<2>,
           index, index) -> !fir.ref<i32>
      %z = fir.array_coor %good [%unit] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%x) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%v) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%z) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // The highest standard Fortran rank must retain every dimension in the
  // flattened address calculation.
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

  // Pointer, allocatable, and polymorphic descriptors use the same existing
  // flattened fast path as a plain box.
  func.func @storage_wrapped_descriptors(
      %heap: !fir.box<!fir.heap<!fir.array<?xi32>>>,
      %pointer: !fir.box<!fir.ptr<!fir.array<?xi32>>>,
      %polymorphic: !fir.class<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %heapAddress = fir.array_coor %heap [%slice] %i
        : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %pointerAddress = fir.array_coor %pointer [%slice] %i
        : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %polymorphicAddress = fir.array_coor %polymorphic [%slice] %i
        : (!fir.class<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%heapAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%pointerAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%polymorphicAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // The accessed box may be loaded through a descriptor slot and then
  // declared again. Producer order must not determine argument recognition.
  func.func @indirect_pointer_descriptor(
      %slot: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
    %declaredSlot = fir.declare %slot {uniq_name = "pointer_slot"}
      : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
        -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %box = fir.load %declaredSlot
      : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %declaredBox = fir.declare %box {uniq_name = "pointer_value"}
      : (!fir.box<!fir.ptr<!fir.array<?xi32>>>)
        -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %declaredBox [%slice] %i
        : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A descriptor loaded inside the loop cannot be hoisted into its guard.
  func.func @loop_local_pointer_descriptor(
      %slot: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %box = fir.load %slot
        : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      %address = fir.array_coor %box [%slice] %i
        : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // The generic array-coordinate lowering widens each operand to index before
  // arithmetic. An i8 coordinate with an i64 section lower must not have its
  // section adjustment truncated back to i8 in the fast clone.
  func.func @narrow_index_wide_section_lower(
      %a: !fir.box<!fir.array<?xi32>>, %narrow: i8, %sectionLower: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %narrow
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i8) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Generic fir.array_coor sign-extends an unsigned section lower's bit
  // pattern to index; the fast clone must not zero-extend it.
  func.func @unsigned_narrow_section_lower(
      %a: !fir.box<!fir.array<?xi32>>, %sectionLower: ui8) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (ui8, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Match generic array-coordinate lowering for a raw i1 coordinate: its set
  // bit is sign-extended through i2 before conversion to index.
  func.func @one_bit_coordinate_sign_extended(
      %a: !fir.box<!fir.array<?xi32>>, %coordinate: i1) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i1) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Both source lower bounds are non-default. Bind each widened coordinate,
  // section adjustment, outer byte stride, and final flattened address.
  func.func @shifted_multidim_narrow_coords(
      %a: !fir.box<!fir.array<?x?xi32>>, %origin0: i8,
      %origin1: index, %section0: i64, %section1: i64,
      %index0: i8, %index1: i16) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape_shift %origin0, %c8, %origin1, %c8
      : (i8, index, index, index) -> !fir.shapeshift<2>
    %slice = fir.slice %section0, %c8, %c1,
                       %section1, %c8, %c1
      : (i64, index, index, i64, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %index0, %index1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shapeshift<2>,
           !fir.slice<2>, i8, i16) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // fir.shift supplies non-default origins. The second slice dimension is
  // Scalar because its upper bound is undefined, even though its step is
  // defined. Its slice lower must not be added to the fast byte offset.
  func.func @shifted_scalar_dimension(
      %a: !fir.box<!fir.array<?x?xi32>>, %origin0: index,
      %origin1: index, %section0: index, %scalarLower: index,
      %index0: index, %index1: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %shape = fir.shift %origin0, %origin1
      : (index, index) -> !fir.shift<2>
    %slice = fir.slice %section0, %c8, %c1,
                       %scalarLower, %undef, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %index0, %index1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.slice<2>,
           index, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A leading Scalar dimension uses its array_coor index but not its slice
  // lower bound. The following Section dimension still contributes its lower
  // bound relative to the non-default source origin.
  func.func @leading_scalar_dimension(
      %a: !fir.box<!fir.array<?x?xi32>>, %origin0: index,
      %origin1: index, %scalarLower: index, %section1: index,
      %index0: index, %index1: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined index
    %shape = fir.shift %origin0, %origin1
      : (index, index) -> !fir.shift<2>
    %slice = fir.slice %scalarLower, %undef, %undef,
                       %section1, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %index0, %index1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.slice<2>,
           index, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A converted fir.undefined is not the scalar-dimension marker recognized
  // by XArrayCoor lowering, so its section lower bound must still contribute.
  func.func @converted_undef_section_dimension(
      %a: !fir.box<!fir.array<?xi32>>, %scalarLower: index,
      %coordinate: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %undef = fir.undefined i32
    %upper = fir.convert %undef : (i32) -> index
    %slice = fir.slice %scalarLower, %upper, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // The module's i2:1 kindmap gives !fir.int<2> a one-bit representation.
  // Its set bit sign-extends to -1 in the generic path, so only the separate
  // ordinary unit-step descriptor may be versioned.
  func.func @one_bit_kind_step_rejected(
      %bad: !fir.box<!fir.array<?xi32>>,
      %good: !fir.box<!fir.array<?xi32>>) {
    %c1_i8 = arith.constant 1 : i8
    %kindStep = fir.convert %c1_i8 : (i8) -> !fir.int<2>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %badSlice = fir.slice %c1, %c8, %kindStep
      : (index, index, !fir.int<2>) -> !fir.slice<1>
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

}

// The core case binds both descriptors to the same guard and verifies the
// typed fast accesses as well as the unchanged sliced fallback.
// CHECK-LABEL: func.func @facerec_core(
// CHECK-SAME: %[[GABOR:.*]]: !fir.box<!fir.array<?x?x?x?xf32>>,
// CHECK-SAME: %[[GRAPH:.*]]: !fir.box<!fir.array<?x?x?xf32>>,
// CHECK-DAG: fir.box_dims %[[GRAPH]],
// CHECK-DAG: fir.box_dims %[[GABOR]],
// CHECK: fir.if
// CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK: } else {
// CHECK: fir.array_coor %[[GRAPH]]{{.*}}[{{.*}}]
// CHECK: fir.array_coor %[[GABOR]]{{.*}}[{{.*}}]

// Section/Scalar/Section preserves the dynamic lower-bound correction.
// CHECK-LABEL: func.func @generalized_shape(
// CHECK: fir.box_dims
// CHECK: fir.if
// CHECK: arith.subi
// CHECK: arith.addi
// CHECK: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK: } else {
// CHECK: fir.array_coor {{.*}}[{{.*}}]

// Distinct and shared fir.slice values both use the common emitter.
// CHECK-LABEL: func.func @two_accesses_one_descriptor(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @two_accesses_shared_slice(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @zero_leading_offset(
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor

// An unsupported sibling rejects only its descriptor in this loop.
// CHECK-LABEL: func.func @same_descriptor_unsupported_sibling(
// CHECK-NOT: fir.if
// CHECK-COUNT-2: fir.array_coor
// CHECK: return

// Mixed flat and sliced accesses are accepted by the common emitter.
// CHECK-LABEL: func.func @mixed_flat_and_slice_supported(
// CHECK: fir.if
// CHECK-COUNT-4: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-3: fir.array_coor
// CHECK: fir.coordinate_of
// CHECK-LABEL: func.func @sibling_flat_and_slice_owners_supported(
// CHECK-COUNT-2: fir.if
// CHECK-LABEL: func.func @wrapped_flat_and_slice_owners_supported(
// CHECK-COUNT-2: fir.if
// CHECK-LABEL: func.func @independent_mixed_owners_supported(
// CHECK-COUNT-2: fir.if

// Unsupported descriptors do not block independent supported descriptors.
// CHECK-LABEL: func.func @descriptor_independence(
// CHECK-SAME: %[[BAD_DESCRIPTOR:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[GOOD_DESCRIPTOR:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK: fir.box_dims %[[GOOD_DESCRIPTOR]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[BAD_DESCRIPTOR]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor

// Unit-step recognition rejects a raw i1 without hiding an independent
// ordinary unit step in the same loop.
// CHECK-LABEL: func.func @i1_steps_rejected(
// CHECK-SAME: %[[RAW_I1:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[ORDINARY_ONE:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK: fir.box_dims %[[ORDINARY_ONE]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[RAW_I1]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor %[[RAW_I1]]
// CHECK: fir.array_coor %[[ORDINARY_ONE]]
// CHECK-LABEL: func.func @signed_one_bit_convert_adjusted(
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK-LABEL: func.func @static_one_lowers_allow_source_widths(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK-LABEL: func.func @static_one_lower_does_not_hide_nonunit_step(
// CHECK: fir.if
// CHECK: fir.array_coor
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @truncated_step_rejected(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK: return
// CHECK-LABEL: func.func @literal_one_convert_chain_step(
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK-LABEL: func.func @noninteger_static_one_step_rejected(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK: return

// Mixed and sequential owners retain loop-local decisions.
// CHECK-LABEL: func.func @slice_with_independent_flat_uses_slice_free_path(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @late_slice_free_rejection_preserves_slice(
// CHECK: fir.if
// CHECK-COUNT-3: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-3: fir.array_coor
// CHECK-LABEL: func.func @independent_owners_supported(
// CHECK-COUNT-2: fir.if
// CHECK-LABEL: func.func @independent_owner_rejection_is_local(
// CHECK: fir.array_coor
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK-LABEL: func.func @independent_owner_late_rejection_is_local(
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK: fir.array_coor

// Nested owners are rewritten through the existing post-order clone walk.
// CHECK-LABEL: func.func @nested_independent_owners_supported(
// CHECK-COUNT-3: fir.if
// CHECK-LABEL: func.func @nested_owners_supported(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK-LABEL: func.func @three_level_owners_supported(
// CHECK: fir.if
// CHECK-COUNT-2: fir.coordinate_of
// CHECK-LABEL: func.func @three_level_late_rejection_blocks_ancestor(
// CHECK-NOT: fir.box_dims
// CHECK-NOT: fir.if
// CHECK-COUNT-2: fir.array_coor
// CHECK: return
// CHECK-LABEL: func.func @nested_slice_flat_supported(
// CHECK: fir.if
// CHECK-COUNT-4: fir.coordinate_of
// CHECK-LABEL: func.func @nested_non_loop_region_supported(
// CHECK: fir.box_dims
// CHECK: fir.if
// CHECK: fir.coordinate_of

// A dynamic step remains generic while the two independent unit slices are
// flattened in the fast clone.
// CHECK-LABEL: func.func @dynamic_step_rejected(
// CHECK-SAME: %[[FIRST:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[DYNAMIC:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[LAST:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-DAG: fir.box_dims %[[FIRST]],
// CHECK-DAG: fir.box_dims %[[LAST]],
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK: fir.array_coor %[[DYNAMIC]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-3: fir.array_coor

// Both wrapper orders contribute a guard and a rewritten fast access.
// CHECK-LABEL: func.func @interleaved_wrapper_order(
// CHECK: %[[GOOD_WRAPPER:.*]] = fir.rebox
// CHECK: %[[BOUNDARY_WRAPPER:.*]] = fir.declare
// CHECK-DAG: fir.box_dims %[[GOOD_WRAPPER]],
// CHECK-DAG: fir.box_dims %[[BOUNDARY_WRAPPER]],
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor %[[GOOD_WRAPPER]]
// CHECK: fir.array_coor %[[BOUNDARY_WRAPPER]]
// CHECK-LABEL: func.func @wrapper_identity_is_preserved(
// CHECK: %[[DECLARED:.*]] = fir.declare
// CHECK: %[[REBOXED:.*]] = fir.rebox %[[DECLARED]]
// CHECK-DAG: fir.box_dims %[[DECLARED]],
// CHECK-DAG: fir.box_dims %[[REBOXED]],
// CHECK: fir.if
// CHECK-DAG: %[[DECLARED_BOX:.*]] = fir.convert %[[DECLARED]]
// CHECK-DAG: %[[REBOXED_BOX:.*]] = fir.convert %[[REBOXED]]
// CHECK-DAG: %[[DECLARED_BASE:.*]] = fir.box_addr %[[DECLARED_BOX]]
// CHECK-DAG: %[[REBOXED_BASE:.*]] = fir.box_addr %[[REBOXED_BOX]]
// CHECK-DAG: fir.coordinate_of %[[DECLARED_BASE]]
// CHECK-DAG: fir.coordinate_of %[[REBOXED_BASE]]
// CHECK-LABEL: func.func @noncontiguous_rebox_rejected(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK: return

// Unsupported slice carriers, component paths, rank changes, and local
// descriptors remain generic without blocking an independent valid access.
// CHECK-LABEL: func.func @fail_closed_slice_forms(
// CHECK-SAME: %[[BAD_TRIPLE:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[BAD_CHARACTER:[^:]+]]: !fir.ref<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME: %[[BAD_RECORD:[^:]+]]: !fir.box<!fir.array<?x!fir.type<slice_element{i:i32}>>>,
// CHECK-SAME: %[[GOOD_FORM:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK: fir.box_dims %[[GOOD_FORM]],
// CHECK: fir.if
// CHECK-COUNT-3: fir.array_coor
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-4: fir.array_coor
// CHECK-LABEL: func.func @component_path_rejected(
// CHECK-SAME: %[[BAD_COMPONENT:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[GOOD_COMPONENT:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK: fir.box_dims %[[GOOD_COMPONENT]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[BAD_COMPONENT]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @trailing_unsupported_slice_triple(
// CHECK-SAME: %[[BAD_TRAILING:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[GOOD_TRAILING:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK: fir.box_dims %[[GOOD_TRAILING]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[BAD_TRAILING]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @reduced_rank_rejected(
// CHECK-SAME: %[[BAD_RANK:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[GOOD_RANK:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK: fir.box_dims %[[GOOD_RANK]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[BAD_RANK]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-2: fir.array_coor
// CHECK-LABEL: func.func @opaque_slice_rejected(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK: return
// CHECK-LABEL: func.func @loop_local_descriptor_rejected(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK: return
// CHECK-LABEL: func.func @scalar_slices_supported(
// CHECK-SAME: %[[ALL_SCALAR:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[LEADING_SCALAR:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[SCALAR_GOOD:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[ORIGIN0:[^:]+]]: index, %[[ORIGIN1:[^:]+]]: index,
// CHECK-SAME: %[[SCALAR_LOWER:[^:]+]]: index, %[[SECTION_LOWER:[^:]+]]: index,
// CHECK-SAME: %[[SCALAR_INDEX0:[^:]+]]: index, %[[SCALAR_INDEX1:[^:]+]]: index)
// CHECK: fir.box_dims %[[LEADING_SCALAR]],
// CHECK: %[[LEADING_DIM1:[^ :]+]]:3 = fir.box_dims %[[LEADING_SCALAR]],
// CHECK: fir.if
// CHECK-DAG: %[[SCALAR_REL1:.*]] = arith.subi %[[SCALAR_INDEX1]], %[[ORIGIN1]] : index
// CHECK-DAG: %[[SCALAR_ADJUST1:.*]] = arith.subi %[[SECTION_LOWER]], %[[ORIGIN1]] : index
// CHECK-DAG: %[[SCALAR_COOR1:.*]] = arith.addi %[[SCALAR_REL1]], %[[SCALAR_ADJUST1]] : index
// CHECK-DAG: %[[SCALAR_BYTES:.*]] = arith.muli %[[LEADING_DIM1]]#2, %[[SCALAR_COOR1]] : index
// CHECK-DAG: %[[SCALAR_REL0:.*]] = arith.subi %[[SCALAR_INDEX0]], %[[ORIGIN0]] : index
// CHECK-DAG: %[[SCALAR_ELEMENTS:.*]] = arith.shrsi %[[SCALAR_BYTES]], {{.*}} : index
// CHECK-DAG: %[[SCALAR_FLAT:.*]] = arith.addi %[[SCALAR_ELEMENTS]], %[[SCALAR_REL0]] : index
// CHECK-DAG: fir.coordinate_of {{.*}}%[[SCALAR_FLAT]]
// CHECK: } else {
// CHECK-COUNT-3: fir.array_coor

// The highest standard Fortran rank retains every descriptor dimension.
// CHECK-LABEL: func.func @rank15_accepted(
// CHECK-COUNT-15: fir.box_dims
// CHECK: fir.if
// CHECK: fir.coordinate_of
// CHECK-LABEL: func.func @loop_local_slice_operand(
// CHECK: fir.if
// CHECK: fir.coordinate_of

// Storage wrappers use the existing flattened emitter.
// CHECK-LABEL: func.func @storage_wrapped_descriptors(
// CHECK: fir.if
// CHECK-COUNT-3: fir.coordinate_of
// CHECK: } else {
// CHECK-COUNT-3: fir.array_coor

// The guard and fast clone must use the loaded descriptor, while the fallback
// keeps its original sliced access. A loop-local load must remain unversioned.
// CHECK-LABEL: func.func @indirect_pointer_descriptor(
// CHECK: %[[BOX:.*]] = fir.load
// CHECK: %[[ACCESS:.*]] = fir.declare %[[BOX]]
// CHECK: fir.box_dims %[[ACCESS]],
// CHECK: fir.if
// CHECK: fir.convert %[[ACCESS]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor %[[ACCESS]] [
// CHECK-LABEL: func.func @loop_local_pointer_descriptor(
// CHECK-NOT: fir.if
// CHECK: fir.array_coor
// CHECK-NOT: fir.if
// CHECK: return

// Canonicalization preserves the same decision: the ordinary one is accepted
// while the raw i1 remains generic.
// CANONICALIZED-LABEL: func.func @i1_steps_rejected(
// CANONICALIZED-SAME: %[[BAD:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CANONICALIZED-SAME: %[[GOOD:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CANONICALIZED-NOT: fir.box_dims %[[BAD]],
// CANONICALIZED: fir.box_dims %[[GOOD]],
// CANONICALIZED: fir.if
// CANONICALIZED: fir.array_coor %[[BAD]]
// CANONICALIZED: fir.coordinate_of
// CANONICALIZED: } else {
// CANONICALIZED: fir.array_coor %[[BAD]]
// CANONICALIZED: fir.array_coor %[[GOOD]]

// The narrow coordinate and wider section lower must each reach index before
// subtraction; a subtraction in i8 would silently truncate the adjustment.
// CHECK-LABEL: func.func @narrow_index_wide_section_lower(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[NARROW:[^:]+]]: i8, %[[LOWER:[^:]+]]: i64)
// CHECK: fir.if
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[NARROW]] : i8 to index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[REL:.*]] = arith.subi %[[INDEX]], %[[ONE]] : index
// CHECK: %[[SECTION:.*]] = arith.index_cast %[[LOWER]] : i64 to index
// CHECK: %[[ADJUST:.*]] = arith.subi %[[SECTION]], %[[ONE]] : index
// CHECK: %[[COOR:.*]] = arith.addi %[[REL]], %[[ADJUST]] : index
// CHECK: fir.coordinate_of {{.*}}%[[COOR]]
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// CHECK-LABEL: func.func @unsigned_narrow_section_lower(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[LOWER:[^:]+]]: ui8)
// CHECK: fir.if
// CHECK: %[[SIGNED:.*]] = fir.convert %[[LOWER]] : (ui8) -> i8
// CHECK: %[[WIDE:.*]] = arith.index_cast %[[SIGNED]] : i8 to index
// CHECK: arith.subi %[[WIDE]], {{.*}} : index
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// A one-bit coordinate follows the generic lowering's signed index
// interpretation before participating in the sliced flat coordinate.
// CHECK-LABEL: func.func @one_bit_coordinate_sign_extended(
// CHECK-SAME: %[[I1_ARRAY:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[I1_COORD:[^:]+]]: i1)
// CHECK: fir.if
// CHECK: %[[I2_COORD:.*]] = arith.extsi %[[I1_COORD]] : i1 to i2
// CHECK: %[[I1_INDEX:.*]] = arith.index_cast %[[I2_COORD]] : i2 to index
// CHECK: %[[I1_REL:.*]] = arith.subi %[[I1_INDEX]], {{.*}} : index
// CHECK: fir.coordinate_of {{.*}}%[[I1_REL]]
// CHECK: } else {
// CHECK: fir.array_coor %[[I1_ARRAY]]
// Shape-shift origins and section lowers are distinct. Both dimension offsets
// must be formed from the matching SSA operands, then combined through the
// descriptor's outer byte stride into the fast flattened element index.
// CHECK-LABEL: func.func @shifted_multidim_narrow_coords(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[ORIGIN0:[^:]+]]: i8, %[[ORIGIN1:[^:]+]]: index,
// CHECK-SAME: %[[SECTION0:[^:]+]]: i64, %[[SECTION1:[^:]+]]: i64,
// CHECK-SAME: %[[INDEX0:[^:]+]]: i8, %[[INDEX1:[^:]+]]: i16)
// CHECK: fir.box_dims %[[A]],
// CHECK: %[[DIM1:[^ :]+]]:3 = fir.box_dims %[[A]],
// CHECK: fir.if
// CHECK: %[[WIDE1:.*]] = arith.index_cast %[[INDEX1]] : i16 to index
// CHECK: %[[REL1:.*]] = arith.subi %[[WIDE1]], %[[ORIGIN1]] : index
// CHECK: %[[LOWER1:.*]] = arith.index_cast %[[SECTION1]] : i64 to index
// CHECK: %[[ADJUST1:.*]] = arith.subi %[[LOWER1]], %[[ORIGIN1]] : index
// CHECK: %[[COOR1:.*]] = arith.addi %[[REL1]], %[[ADJUST1]] : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[DIM1]]#2, %[[COOR1]] : index
// CHECK: %[[WIDE0:.*]] = arith.index_cast %[[INDEX0]] : i8 to index
// CHECK: %[[ORIGIN0_INDEX:.*]] = arith.index_cast %[[ORIGIN0]] : i8 to index
// CHECK: %[[REL0:.*]] = arith.subi %[[WIDE0]], %[[ORIGIN0_INDEX]] : index
// CHECK: %[[LOWER0:.*]] = arith.index_cast %[[SECTION0]] : i64 to index
// CHECK: %[[ADJUST0:.*]] = arith.subi %[[LOWER0]], %[[ORIGIN0_INDEX]] : index
// CHECK: %[[COOR0:.*]] = arith.addi %[[REL0]], %[[ADJUST0]] : index
// CHECK: %[[ELEMENTS:.*]] = arith.shrsi %[[BYTES]], {{.*}} : index
// CHECK: %[[FLAT:.*]] = arith.addi %[[ELEMENTS]], %[[COOR0]] : index
// CHECK: fir.coordinate_of {{.*}}%[[FLAT]]
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// A defined step does not turn a Scalar triple into a Section. The dim-1
// contribution must be the shifted coordinate itself, with no slice lower.
// CHECK-LABEL: func.func @shifted_scalar_dimension(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[ORIGIN0:[^:]+]]: index, %[[ORIGIN1:[^:]+]]: index,
// CHECK-SAME: %[[SECTION0:[^:]+]]: index, %[[SCALAR_LOWER:[^:]+]]: index,
// CHECK-SAME: %[[INDEX0:[^:]+]]: index, %[[INDEX1:[^:]+]]: index)
// CHECK: fir.box_dims %[[A]],
// CHECK: %[[DIM1:[^ :]+]]:3 = fir.box_dims %[[A]],
// CHECK: fir.if
// CHECK: %[[SCALAR_REL:.*]] = arith.subi %[[INDEX1]], %[[ORIGIN1]] : index
// CHECK-NOT: arith.subi %[[SCALAR_LOWER]], %[[ORIGIN1]] : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[DIM1]]#2, %[[SCALAR_REL]] : index
// CHECK: %[[REL0:.*]] = arith.subi %[[INDEX0]], %[[ORIGIN0]] : index
// CHECK: %[[ADJUST0:.*]] = arith.subi %[[SECTION0]], %[[ORIGIN0]] : index
// CHECK: %[[COOR0:.*]] = arith.addi %[[REL0]], %[[ADJUST0]] : index
// CHECK: %[[ELEMENTS:.*]] = arith.shrsi %[[BYTES]], {{.*}} : index
// CHECK: %[[FLAT:.*]] = arith.addi %[[ELEMENTS]], %[[COOR0]] : index
// CHECK: fir.coordinate_of {{.*}}%[[FLAT]]
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// A leading Scalar dimension omits its slice lower while the following
// Section dimension retains its lower-bound adjustment.
// CHECK-LABEL: func.func @leading_scalar_dimension(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME: %[[ORIGIN0:[^:]+]]: index, %[[ORIGIN1:[^:]+]]: index,
// CHECK-SAME: %[[SCALAR_LOWER:[^:]+]]: index, %[[SECTION1:[^:]+]]: index,
// CHECK-SAME: %[[INDEX0:[^:]+]]: index, %[[INDEX1:[^:]+]]: index)
// CHECK: fir.box_dims %[[A]],
// CHECK: %[[DIM1:[^ :]+]]:3 = fir.box_dims %[[A]],
// CHECK: fir.if
// CHECK: %[[REL1:.*]] = arith.subi %[[INDEX1]], %[[ORIGIN1]] : index
// CHECK: %[[ADJUST1:.*]] = arith.subi %[[SECTION1]], %[[ORIGIN1]] : index
// CHECK: %[[COOR1:.*]] = arith.addi %[[REL1]], %[[ADJUST1]] : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[DIM1]]#2, %[[COOR1]] : index
// CHECK: %[[REL0:.*]] = arith.subi %[[INDEX0]], %[[ORIGIN0]] : index
// CHECK-NOT: arith.subi %[[SCALAR_LOWER]], %[[ORIGIN0]] : index
// CHECK: %[[ELEMENTS:.*]] = arith.shrsi %[[BYTES]], {{.*}} : index
// CHECK: %[[FLAT:.*]] = arith.addi %[[ELEMENTS]], %[[REL0]] : index
// CHECK: fir.coordinate_of {{.*}}%[[FLAT]]
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// A converted undefined upper bound remains a section, so its lower bound
// must contribute to the fast address before and after canonicalization.
// CHECK-LABEL: func.func @converted_undef_section_dimension(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[SCALAR_LOWER:[^:]+]]: index, %[[COORD:[^:]+]]: index)
// CHECK: fir.if
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[REL:.*]] = arith.subi %[[COORD]], %[[ONE]] : index
// CHECK: %[[ADJUST:.*]] = arith.subi %[[SCALAR_LOWER]], %[[ONE]] : index
// CHECK: %[[COOR:.*]] = arith.addi %[[REL]], %[[ADJUST]] : index
// CHECK: fir.coordinate_of {{.*}}%[[COOR]]
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]

// CANONICALIZED-LABEL: func.func @converted_undef_section_dimension(
// CANONICALIZED-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CANONICALIZED-SAME: %[[LOWER:[^:]+]]: index, %[[COORD:[^:]+]]: index)
// CANONICALIZED: fir.if
// CANONICALIZED: %[[ONE:.*]] = arith.constant 1 : index
// CANONICALIZED: %[[REL:.*]] = arith.subi %[[COORD]], %[[ONE]] : index
// CANONICALIZED: %[[ADJUST:.*]] = arith.subi %[[LOWER]], %[[ONE]] : index
// CANONICALIZED: %[[COOR:.*]] = arith.addi %[[REL]], %[[ADJUST]] : index
// CANONICALIZED: fir.coordinate_of {{.*}}%[[COOR]]
// CANONICALIZED: } else {
// CANONICALIZED: fir.array_coor %[[A]]

// POST-CANON-LABEL: func.func @converted_undef_section_dimension(
// POST-CANON-SAME: %[[A:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// POST-CANON-SAME: %[[SCALAR_LOWER:[^:]+]]: index, %[[COORD:[^:]+]]: index)
// POST-CANON: %[[ONE:.*]] = arith.constant 1 : index
// POST-CANON: fir.if
// POST-CANON: %[[REL:.*]] = arith.subi %[[COORD]], %[[ONE]] : index
// POST-CANON: %[[ADJUST:.*]] = arith.subi %[[SCALAR_LOWER]], %[[ONE]] : index
// POST-CANON: %[[COOR:.*]] = arith.addi %[[REL]], %[[ADJUST]] : index
// POST-CANON: fir.coordinate_of {{.*}}%[[COOR]]
// POST-CANON: } else {
// POST-CANON: fir.array_coor %[[A]]

// CHECK-LABEL: func.func @one_bit_kind_step_rejected(
// CHECK-SAME: %[[BAD:[^:]+]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME: %[[GOOD:[^:]+]]: !fir.box<!fir.array<?xi32>>)
// CHECK-NOT: fir.box_dims %[[BAD]],
// CHECK: fir.box_dims %[[GOOD]],
// CHECK: fir.if
// CHECK: fir.array_coor %[[BAD]]
// CHECK: fir.coordinate_of
// CHECK: } else {
// CHECK: fir.array_coor %[[BAD]]
// CHECK: fir.array_coor %[[GOOD]]

// Exact guard-query counts pair with the descriptor bindings above. Together
// they prove that accepted descriptors are queried and rejected ones are not,
// independently of DenseMap traversal order.
// GUARDS-LABEL: func.func @descriptor_independence(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @i1_steps_rejected(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @static_one_lower_does_not_hide_nonunit_step(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @dynamic_step_rejected(
// GUARDS-COUNT-2: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @wrapper_identity_is_preserved(
// GUARDS-COUNT-2: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @fail_closed_slice_forms(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @component_path_rejected(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @trailing_unsupported_slice_triple(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @reduced_rank_rejected(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @scalar_slices_supported(
// GUARDS-COUNT-5: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @one_bit_coordinate_sign_extended(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if
// GUARDS-LABEL: func.func @one_bit_kind_step_rejected(
// GUARDS-COUNT-1: fir.box_dims
// GUARDS-NOT: fir.box_dims
// GUARDS: fir.if

// CANON-GUARDS-LABEL: func.func @i1_steps_rejected(
// CANON-GUARDS-COUNT-1: fir.box_dims
// CANON-GUARDS-NOT: fir.box_dims
// CANON-GUARDS: fir.if

// Disabling slices leaves even an otherwise supported sliced owner unchanged.
// DISABLED-LABEL: func.func @narrow_index_wide_section_lower(
// DISABLED-NOT: fir.if
// DISABLED: fir.array_coor
// DISABLED: return
