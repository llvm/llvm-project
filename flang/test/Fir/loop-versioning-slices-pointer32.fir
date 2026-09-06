// RUN: fir-opt --loop-versioning --cfg-conversion \
// RUN:   --cg-rewrite \
// RUN:   --fir-to-llvm-ir="target=i386-unknown-linux-gnu" \
// RUN:   --reconcile-unrealized-casts %s -o %t
// RUN: FileCheck %s --input-file=%t --enable-var-scope
// RUN: FileCheck %s --input-file=%t --check-prefix=P32-ARITH \
// RUN:   --enable-var-scope

// Slices use the effective 32-bit address domain on a p32 target.
// Narrow signed operands are extended before any offset arithmetic; the
// generic fallback retains XArrayCoor's established i64 arithmetic.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:32:32:32:32"
} {
  func.func private @use(!fir.ref<i32>)

  func.func @slice_pointer32(
      %a: !fir.box<!fir.array<?x?xi32>>, %innerLower: i16,
      %outerLower: ui32, %outer: i32) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %shape = fir.shape %c8, %c8 : (index, index) -> !fir.shape<2>
    %slice = fir.slice %innerLower, %c8, %c1,
                       %outerLower, %c8, %c1
      : (i16, index, index, ui32, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a(%shape) [%slice] %i, %outer
        : (!fir.box<!fir.array<?x?xi32>>, !fir.shape<2>, !fir.slice<2>,
           index, i32) -> !fir.ref<i32>
      fir.call @use(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // Every operand consumed by the p32 byte path must be representable without
  // changing the generic XArrayCoor interpretation.
  func.func @slice_pointer32_rejected_operands(
      %wideIndexArg: !fir.box<!fir.array<?xi32>>,
      %wideLowerArg: !fir.box<!fir.array<?xi32>>,
      %unsignedLowerArg: !fir.box<!fir.array<?xi32>>,
      %unsignedWideArg: !fir.box<!fir.array<?xi32>>,
      %wideIndex: i64, %wideLower: i64, %unsignedLower: ui16,
      %unsignedWide: ui64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %unit = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    %wideLowerSlice = fir.slice %wideLower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    %unsignedLowerSlice = fir.slice %unsignedLower, %c8, %c1
      : (ui16, index, index) -> !fir.slice<1>
    %unsignedWideSlice = fir.slice %unsignedWide, %c8, %c1
      : (ui64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %wideIndexAddress = fir.array_coor %wideIndexArg [%unit] %wideIndex
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i64)
          -> !fir.ref<i32>
      %wideLowerAddress =
          fir.array_coor %wideLowerArg [%wideLowerSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %unsignedLowerAddress =
          fir.array_coor %unsignedLowerArg [%unsignedLowerSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      %unsignedWideAddress =
          fir.array_coor %unsignedWideArg [%unsignedWideSlice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use(%wideIndexAddress) : (!fir.ref<i32>) -> ()
      fir.call @use(%wideLowerAddress) : (!fir.ref<i32>) -> ()
      fir.call @use(%unsignedLowerAddress) : (!fir.ref<i32>) -> ()
      fir.call @use(%unsignedWideAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A proven unit section lower is not materialized in the p32 fast path, so
  // its wider source type does not restrict versioning.
  func.func @slice_pointer32_static_wide_one(
      %a: !fir.box<!fir.array<?xi32>>) {
    %wideOne = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %wideOne, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

}

// CHECK-LABEL: llvm.func @slice_pointer32(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr
// CHECK-SAME: %[[INNER16:[^:]+]]: i16
// CHECK-SAME: %[[OUTER_LOWER:[^:]+]]: i32
// CHECK-SAME: %[[OUTER:[^:]+]]: i32
// CHECK: %[[D0_INDEX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[D0_STRIDE_PTR:.*]] = llvm.getelementptr %[[A]][0, 7, %[[D0_INDEX]], 2]
// CHECK: %[[D0_STRIDE:.*]] = llvm.load %[[D0_STRIDE_PTR]] : !llvm.ptr -> i32
// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: llvm.icmp "eq" %[[D0_STRIDE]], %[[SIZE]] : i32
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[D1_INDEX:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[D1_STRIDE_PTR:.*]] = llvm.getelementptr %[[A]][0, 7, %[[D1_INDEX]], 2]
// CHECK: %[[D1_STRIDE:.*]] = llvm.load %[[D1_STRIDE_PTR]] : !llvm.ptr -> i32
// CHECK: %[[BASE_PTR:.*]] = llvm.getelementptr %[[A]][0, 0]
// CHECK: %[[BYTE_BASE:.*]] = llvm.load %[[BASE_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[INNER_INDEX_ADJUST:.*]] = llvm.sub {{.*}}, %[[FAST_ONE]] : i32
// CHECK: %[[INNER32:.*]] = llvm.sext %[[INNER16]] : i16 to i32
// CHECK: %[[INNER_LOWER_ADJUST:.*]] = llvm.sub %[[INNER32]], %[[FAST_ONE]] : i32
// CHECK: %[[INNER_OFFSET:.*]] = llvm.add %[[INNER_INDEX_ADJUST]], %[[INNER_LOWER_ADJUST]] : i32
// CHECK: %[[INNER_BYTES:.*]] = llvm.mul %[[INNER_OFFSET]], %[[SIZE]] : i32
// CHECK: %[[OUTER_INDEX_ADJUST:.*]] = llvm.sub %[[OUTER]], %[[FAST_ONE]] : i32
// CHECK: %[[OUTER_LOWER_ADJUST:.*]] = llvm.sub %[[OUTER_LOWER]], %[[FAST_ONE]] : i32
// CHECK: %[[OUTER_OFFSET:.*]] = llvm.add %[[OUTER_INDEX_ADJUST]], %[[OUTER_LOWER_ADJUST]] : i32
// CHECK: %[[OUTER_BYTES:.*]] = llvm.mul %[[D1_STRIDE]], %[[OUTER_OFFSET]] : i32
// CHECK: %[[TOTAL:.*]] = llvm.add %[[INNER_BYTES]], %[[OUTER_BYTES]] : i32
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr %[[BYTE_BASE]][%[[TOTAL]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: llvm.call @use(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: llvm.sext %[[INNER16]] : i16 to i64
// CHECK: llvm.sext %[[OUTER]] : i32 to i64
// CHECK: llvm.sext %[[OUTER_LOWER]] : i32 to i64
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use(%[[FALLBACK_ADDR]])

// CHECK-LABEL: llvm.func @slice_pointer32_rejected_operands(
// CHECK-SAME: %[[WIDE_INDEX:[^:]+]]: i64
// CHECK-SAME: %[[WIDE_LOWER:[^:]+]]: i64
// CHECK-SAME: %[[UNSIGNED_LOWER:[^:]+]]: i16
// CHECK-SAME: %[[UNSIGNED_WIDE:[^:]+]]: i64
// CHECK-COUNT-1: llvm.cond_br
// CHECK-NOT: llvm.cond_br
// CHECK-NOT: (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[WIDE_INDEX_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: %[[WIDE_LOWER_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: %[[UNSIGNED_LOWER_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: %[[UNSIGNED_WIDE_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use(%[[WIDE_INDEX_ADDR]])
// CHECK: llvm.call @use(%[[WIDE_LOWER_ADDR]])
// CHECK: llvm.call @use(%[[UNSIGNED_LOWER_ADDR]])
// CHECK: llvm.call @use(%[[UNSIGNED_WIDE_ADDR]])
// CHECK-NOT: llvm.cond_br

// CHECK-LABEL: llvm.func @slice_pointer32_static_wide_one(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK-NOT: llvm.trunc
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: llvm.call @use(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use(%[[FALLBACK_ADDR]])

// P32-ARITH-LABEL: llvm.func @slice_pointer32(
// P32-ARITH: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// P32-ARITH: ^[[FAST]]:
// P32-ARITH-NOT: llvm.ashr
// P32-ARITH-NOT: llvm.lshr
// P32-ARITH-NOT: llvm.shl
// P32-ARITH-NOT: llvm.sdiv
// P32-ARITH-NOT: llvm.udiv
// P32-ARITH: ^[[FALLBACK]]:
