// RUN: fir-opt --loop-versioning %s -o %t.enabled
// RUN: fir-opt '--loop-versioning=enable-slices=false' %s \
// RUN:   -o %t.disabled
// RUN: diff %t.enabled %t.disabled
// RUN: FileCheck %s --check-prefix=RESULT --input-file=%t.enabled \
// RUN:   --enable-var-scope

// Compare emitted FIR byte-for-byte to verify that the enable-slices option
// does not affect slice-free or rejected-descriptor transformations.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:64:64:64:64"
} {
  func.func private @use_i32(!fir.ref<i32>)

  // A slice-free candidate must retain the existing versioned form when slice
  // support is enabled, since the feature is enabled by default.
  func.func @slice_free(
      %a: !fir.box<!fir.array<?x?xi32>>, %outer: index,
      %initial: i32) -> i32 {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %result = fir.do_loop %i = %c1 to %c8 step %c1
        iter_args(%sum = %initial) -> (i32) {
      %address = fir.array_coor %a %i, %outer
        : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
      %value = fir.load %address : !fir.ref<i32>
      %next = arith.addi %sum, %value : i32
      fir.result %next : i32
    }
    return %result : i32
  }

  // RESULT-LABEL: func.func @slice_free(
  // RESULT: %[[IF:.*]] = fir.if
  // RESULT: fir.coordinate_of
  // RESULT: } else {
  // RESULT: fir.array_coor
  // RESULT: return %[[IF]] : i32

  // The slice switch does not undo support for a pointer dummy passed by box
  // address. Its loaded descriptor is versioned in both configurations.
  func.func @slice_free_pointer_dummy(
      %slot: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
    %box = fir.load %slot
      : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %box %i
        : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // RESULT-LABEL: func.func @slice_free_pointer_dummy(
  // RESULT: fir.if
  // RESULT: fir.coordinate_of
  // RESULT: } else {
  // RESULT: fir.array_coor

  // Interleaved declare/rebox wrappers are normalized for flat accesses too;
  // disabling sliced accesses must not restore the older matcher order.
  func.func @slice_free_interleaved_wrappers(
      %a: !fir.box<!fir.array<?xi32>>) {
    %reboxed = fir.rebox %a
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %declared = fir.declare %reboxed {uniq_name = "interleaved"}
      : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %declared %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // RESULT-LABEL: func.func @slice_free_interleaved_wrappers(
  // RESULT: fir.if
  // RESULT: fir.coordinate_of
  // RESULT: } else {
  // RESULT: fir.array_coor

  // A rejected inner slice must prevent rewriting the flat access to the same
  // descriptor both with slice support enabled and explicitly disabled.
  func.func @nested_rejected_slice_nfc(
      %a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c2
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %outer = fir.array_coor %a %i
        : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      fir.call @use_i32(%outer) : (!fir.ref<i32>) -> ()
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %inner = fir.array_coor %a [%slice] %j
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%inner) : (!fir.ref<i32>) -> ()
      }
    }
    return
  }

  // RESULT-LABEL: func.func @nested_rejected_slice_nfc(
  // RESULT-SAME: %[[A:[^:]+]]: !fir.box
  // RESULT: %[[SLICE:.*]] = fir.slice
  // RESULT-NOT: fir.if
  // RESULT: fir.do_loop
  // RESULT: %[[OUTER:.*]] = fir.array_coor %[[A]]
  // RESULT: fir.call @use_i32(%[[OUTER]])
  // RESULT: fir.do_loop
  // RESULT: %[[INNER:.*]] = fir.array_coor %[[A]]{{.*}}[%[[SLICE]]]
  // RESULT: fir.call @use_i32(%[[INNER]])
  // RESULT-NOT: fir.if
  // RESULT: return
}
