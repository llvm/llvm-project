// RUN: fir-opt --loop-versioning --canonicalize --cfg-conversion \
// RUN:   --cg-rewrite \
// RUN:   --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" \
// RUN:   --reconcile-unrealized-casts --split-input-file %s | \
// RUN:   FileCheck %s --enable-var-scope
// RUN: fir-opt --loop-versioning --split-input-file %s | \
// RUN:   FileCheck %s --check-prefix=FIR --enable-var-scope

// Different power-of-two element sizes in one owner must retain independent
// stride predicates and element scales in the common typed emitter.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 64>
  >,
  fir.defaultkind = "a1c4d8i4l4r4",
  llvm.data_layout = "e-p:64:64"
} {
  func.func private @use_f64(!fir.ref<f64>)
  func.func private @use_i32(!fir.ref<i32>)

  func.func @different_element_sizes(
      %a: !fir.class<!fir.array<?xf64>>,
      %b: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %aAddress = fir.array_coor %a [%slice] %i
        : (!fir.class<!fir.array<?xf64>>, !fir.slice<1>, index)
          -> !fir.ref<f64>
      %bAddress = fir.array_coor %b [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_f64(%aAddress) : (!fir.ref<f64>) -> ()
      fir.call @use_i32(%bAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Pointer and allocatable boxes are converted to typed flattened boxes and
  // lower both fast addresses to descriptor-based GEPs.
  func.func @storage_wrapped_descriptors(
      %heap: !fir.box<!fir.heap<!fir.array<?xi32>>>,
      %pointer: !fir.box<!fir.ptr<!fir.array<?xi32>>>) {
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
      fir.call @use_i32(%heapAddress) : (!fir.ref<i32>) -> ()
      fir.call @use_i32(%pointerAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // The fast path reinterprets narrow unsigned bits as signed before widening,
  // matching the generic array-coordinate path's sign extension.
  func.func @narrow_unsigned_lower(
      %a: !fir.box<!fir.array<?xi32>>, %sectionLower: ui16,
      %coordinate: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (ui16, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i64)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A negative signed section lower is widened before the offset arithmetic
  // and remains tied to the same fast and fallback consumers.
  func.func @negative_section_lower(%a: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cm3 = arith.constant -3 : i64
    %slice = fir.slice %cm3, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Exercise the complete rank-two flattening formula with distinct source
  // origins and section lower bounds in both dimensions.
  func.func @shifted_rank2(
      %a: !fir.box<!fir.array<?x?xi32>>, %origin0: index,
      %origin1: index, %section0: index, %section1: index,
      %index0: index, %index1: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape_shift %origin0, %c8, %origin1, %c8
      : (index, index, index, index) -> !fir.shapeshift<2>
    %slice = fir.slice %section0, %c8, %c1,
                       %section1, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %index0, %index1
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shapeshift<2>,
           !fir.slice<2>, index, index) -> !fir.ref<i32>
      fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
}

// The FIR checks prove that the common emitter keeps typed bases in the fast
// clone and the original sliced accesses in the fallback.
// FIR-LABEL: func.func @different_element_sizes(
// FIR: fir.if
// FIR: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// FIR: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FIR: } else {
// FIR-COUNT-2: fir.array_coor
// FIR-LABEL: func.func @storage_wrapped_descriptors(
// FIR: fir.if
// FIR-COUNT-2: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FIR: } else {
// FIR-COUNT-2: fir.array_coor
// FIR-LABEL: func.func @narrow_unsigned_lower(
// FIR: fir.if
// FIR: fir.convert {{.*}} : (ui16) -> i16
// FIR: arith.index_cast {{.*}} : i16 to index
// FIR: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FIR: } else {
// FIR: fir.array_coor
// FIR-LABEL: func.func @negative_section_lower(
// FIR: fir.if
// FIR: arith.subi %{{.*}}, %{{.*}} : index
// FIR: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FIR: } else {
// FIR: fir.array_coor
// FIR-LABEL: func.func @shifted_rank2(
// FIR: fir.if
// FIR: arith.muli
// FIR: arith.shrsi
// FIR: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FIR: } else {
// FIR: fir.array_coor

// The LLVM checks prove that each descriptor uses its own element size and
// that fast GEPs remain typed for every supported descriptor wrapper.
// CHECK-LABEL: llvm.func @different_element_sizes(
// CHECK-SAME: %[[F64_DESC:[^:]+]]: !llvm.ptr, %[[I32_DESC:[^:]+]]: !llvm.ptr
// CHECK-DAG: %[[F64_STRIDE_PTR:.*]] = llvm.getelementptr %[[F64_DESC]][0, 7, %{{.*}}, 2]
// CHECK-DAG: %[[F64_STRIDE:.*]] = llvm.load %[[F64_STRIDE_PTR]]
// CHECK-DAG: %[[F64_OK:.*]] = llvm.icmp "eq" %[[F64_STRIDE]], %[[SIZE8:[^ ]+]] : i64
// CHECK-DAG: %[[SIZE8]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG: %[[I32_STRIDE_PTR:.*]] = llvm.getelementptr %[[I32_DESC]][0, 7, %{{.*}}, 2]
// CHECK-DAG: %[[I32_STRIDE:.*]] = llvm.load %[[I32_STRIDE_PTR]]
// CHECK-DAG: %[[I32_OK:.*]] = llvm.icmp "eq" %[[I32_STRIDE]], %[[SIZE4:[^ ]+]] : i64
// CHECK-DAG: %[[SIZE4]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NOT: llvm.and %[[I32_OK]], %[[I32_OK]]
// CHECK-NOT: llvm.and %[[F64_OK]], %[[F64_OK]]
// CHECK: llvm.and %{{.*}}, %{{.*}} : i1
// CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.call @use_f64
// CHECK: llvm.call @use_i32

// CHECK-LABEL: llvm.func @storage_wrapped_descriptors(
// CHECK: llvm.cond_br
// CHECK-COUNT-2: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.call @use_i32
// CHECK: llvm.call @use_i32

// XArrayCoor sign-extends the bit pattern of even an unsigned lower bound.
// The fast offset must use the same conversion before its typed GEP.
// CHECK-LABEL: llvm.func @narrow_unsigned_lower(
// CHECK-SAME: %{{.*}}: !llvm.ptr, %[[LOWER:[^:]+]]: i16, %[[COORD:[^:]+]]: i64
// CHECK: %[[INDEX:.*]] = llvm.sub %[[COORD]], %{{.*}} : i64
// CHECK: %[[WIDE_LOWER:.*]] = llvm.sext %[[LOWER]] : i16 to i64
// CHECK: %[[ADJUST:.*]] = llvm.sub %[[WIDE_LOWER]], %{{.*}} : i64
// CHECK: %[[FLAT:.*]] = llvm.add %[[INDEX]], %[[ADJUST]] : i64
// CHECK: llvm.getelementptr %{{.*}}[%[[FLAT]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-LABEL: llvm.func @negative_section_lower(
// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(-5 : i64) : i64
// CHECK: %[[FLAT:.*]] = llvm.add %{{.*}}, %[[OFFSET]] : i64
// CHECK: llvm.getelementptr %{{.*}}[%[[FLAT]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-LABEL: llvm.func @shifted_rank2(
// CHECK-SAME: %[[DESC:[^:]+]]: !llvm.ptr, %[[ORIGIN0:[^:]+]]: i64,
// CHECK-SAME: %[[ORIGIN1:[^:]+]]: i64, %[[SECTION0:[^:]+]]: i64,
// CHECK-SAME: %[[SECTION1:[^:]+]]: i64, %[[INDEX0:[^:]+]]: i64,
// CHECK-SAME: %[[INDEX1:[^:]+]]: i64)
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[STRIDE_PTR:.*]] = llvm.getelementptr %[[DESC]][0, 7, %[[ONE]], 2]
// CHECK: %[[OUTER_STRIDE:.*]] = llvm.load %[[STRIDE_PTR]]
// CHECK: llvm.cond_br
// CHECK: %[[REL1:.*]] = llvm.sub %[[INDEX1]], %[[ORIGIN1]] : i64
// CHECK: %[[ADJUST1:.*]] = llvm.sub %[[SECTION1]], %[[ORIGIN1]] : i64
// CHECK: %[[COOR1:.*]] = llvm.add %[[REL1]], %[[ADJUST1]] : i64
// CHECK: %[[BYTES:.*]] = llvm.mul %[[OUTER_STRIDE]], %[[COOR1]] : i64
// CHECK: %[[REL0:.*]] = llvm.sub %[[INDEX0]], %[[ORIGIN0]] : i64
// CHECK: %[[ADJUST0:.*]] = llvm.sub %[[SECTION0]], %[[ORIGIN0]] : i64
// CHECK: %[[COOR0:.*]] = llvm.add %[[REL0]], %[[ADJUST0]] : i64
// CHECK: %[[ELEMENTS:.*]] = llvm.ashr %[[BYTES]], %{{.*}} : i64
// CHECK: %[[FLAT:.*]] = llvm.add %[[ELEMENTS]], %[[COOR0]] : i64
// CHECK: llvm.getelementptr %{{.*}}[%[[FLAT]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
