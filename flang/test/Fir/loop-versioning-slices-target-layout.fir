// RUN: fir-opt --loop-versioning --cfg-conversion \
// RUN:   --cg-rewrite \
// RUN:   --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" \
// RUN:   --reconcile-unrealized-casts --split-input-file %s | \
// RUN:   FileCheck %s --enable-var-scope
// RUN: fir-opt --loop-versioning --split-input-file %s | \
// RUN:   FileCheck %s --check-prefix=FIR --enable-var-scope

// f80:32 has a 12-byte allocated element size. The rank-two case combines it
// with a runtime outer byte stride, yet slice rewriting stays in the
// byte domain and needs neither a power-of-two size nor a quotient operation.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 64>,
    #dlti.dl_entry<f80, dense<[32, 32]> : vector<2xi64>>
  >,
  fir.defaultkind = "a1c4d8i4l4r4",
  fir.kindmap = "i2:1",
  llvm.data_layout = "e-p:64:64-f80:32"
} {
  func.func private @use_f80(!fir.ref<f80>)
  func.func private @use_i32(!fir.ref<i32>)

  func.func @element_size_12(%a: !fir.class<!fir.array<?x?xf80>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1, %c1, %c8, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      fir.do_loop %j = %c1 to %c8 step %c1 {
        %address = fir.array_coor %a [%slice] %i, %j
          : (!fir.class<!fir.array<?x?xf80>>, !fir.slice<2>, index, index)
            -> !fir.ref<f80>
        fir.call @use_f80(%address) : (!fir.ref<f80>) -> ()
      }
    }
    return
  }

  // One owner may version independent descriptors whose element sizes differ.
  // Each descriptor must retain its own stride predicate and byte scale.
  func.func @different_element_sizes(
      %a: !fir.class<!fir.array<?xf80>>,
      %b: !fir.box<!fir.array<?xi32>>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %aAddress = fir.array_coor %a [%slice] %i
        : (!fir.class<!fir.array<?xf80>>, !fir.slice<1>, index)
          -> !fir.ref<f80>
      %bAddress = fir.array_coor %b [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
      fir.call @use_f80(%aAddress) : (!fir.ref<f80>) -> ()
      fir.call @use_i32(%bAddress) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // An exact target-width unsigned section lower preserves all address bits,
  // while a signless i64 coordinate observes the p64 index domain directly.
  func.func @exact_width_unsigned_lower(
      %a: !fir.box<!fir.array<?xi32>>, %sectionLower: ui64,
      %coordinate: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (ui64, index, index) -> !fir.slice<1>
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

  // The custom FIR integer kind is signed and lowers to i1. Its set bit must
  // therefore be sign-extended to -1 rather than treated as a positive one.
  func.func @kind_mapped_one_bit_widening(%value: i32) -> i64 {
    %bit = fir.convert %value : (i32) -> !fir.int<2>
    %wide = fir.convert %bit : (!fir.int<2>) -> i64
    return %wide : i64
  }
}

// CHECK-LABEL: llvm.func @element_size_12(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr
// CHECK: llvm.cond_br
// CHECK: %[[D0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[D0_STRIDE_ADDR:.*]] = llvm.getelementptr %[[A]][0, 7, %[[D0]], 2]
// CHECK: %[[D0_STRIDE:.*]] = llvm.load %[[D0_STRIDE_ADDR]] : !llvm.ptr -> i64
// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(12 : i64) : i64
// CHECK: llvm.icmp "eq" %[[D0_STRIDE]], %[[SIZE]] : i64
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK-NOT: llvm.{{(sdiv|udiv|shl|ashr|lshr)}}
// CHECK: %[[FAST_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[D1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[D1_STRIDE_ADDR:.*]] = llvm.getelementptr %[[A]][0, 7, %[[D1]], 2]
// CHECK: %[[D1_STRIDE:.*]] = llvm.load %[[D1_STRIDE_ADDR]] : !llvm.ptr -> i64
// CHECK: %[[FAST_D0:.*]] = llvm.mul {{.*}}, %[[SIZE]] : i64
// CHECK: %[[FAST_D1:.*]] = llvm.mul %[[D1_STRIDE]], {{.*}} : i64
// CHECK: %[[FAST_BYTES:.*]] = llvm.add %[[FAST_D0]], %[[FAST_D1]] : i64
// CHECK-NOT: llvm.{{(sdiv|udiv|shl|ashr|lshr)}}
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}}[%[[FAST_BYTES]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NOT: llvm.{{(sdiv|udiv|shl|ashr|lshr)}}
// CHECK: llvm.call @use_f80(%[[FAST_ADDR]])
// CHECK-NOT: llvm.{{(sdiv|udiv|shl|ashr|lshr)}}
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_f80(%[[FALLBACK_ADDR]])

// CHECK-LABEL: llvm.func @different_element_sizes(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr
// CHECK-SAME: %[[B:[^:]+]]: !llvm.ptr
// CHECK: %[[A_D0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[A_STRIDE_ADDR:.*]] = llvm.getelementptr %[[A]][0, 7, %[[A_D0]], 2]
// CHECK: %[[A_STRIDE:.*]] = llvm.load %[[A_STRIDE_ADDR]] : !llvm.ptr -> i64
// CHECK: %[[A_SIZE:.*]] = llvm.mlir.constant(12 : i64) : i64
// CHECK: %[[A_CONTIGUOUS:.*]] = llvm.icmp "eq" %[[A_STRIDE]], %[[A_SIZE]] : i64
// CHECK: %[[B_D0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B_STRIDE_ADDR:.*]] = llvm.getelementptr %[[B]][0, 7, %[[B_D0]], 2]
// CHECK: %[[B_STRIDE:.*]] = llvm.load %[[B_STRIDE_ADDR]] : !llvm.ptr -> i64
// CHECK: %[[B_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[B_CONTIGUOUS:.*]] = llvm.icmp "eq" %[[B_STRIDE]], %[[B_SIZE]] : i64
// CHECK: %[[BOTH:.*]] = llvm.and %[[B_CONTIGUOUS]], %[[A_CONTIGUOUS]] : i1
// CHECK: llvm.cond_br %[[BOTH]], ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[A_BASE_ADDR:.*]] = llvm.getelementptr %[[A]][0, 0]
// CHECK-NEXT: %[[A_BASE:.*]] = llvm.load %[[A_BASE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[B_BASE_ADDR:.*]] = llvm.getelementptr %[[B]][0, 0]
// CHECK-NEXT: %[[B_BASE:.*]] = llvm.load %[[B_BASE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[A_OFFSET:.*]] = llvm.sub {{.*}} : i64
// CHECK: %[[A_BYTES:.*]] = llvm.mul %[[A_OFFSET]], %[[A_SIZE]] : i64
// CHECK: %[[A_FAST:.*]] = llvm.getelementptr %[[A_BASE]][%[[A_BYTES]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: %[[B_OFFSET:.*]] = llvm.sub {{.*}} : i64
// CHECK: %[[B_BYTES:.*]] = llvm.mul %[[B_OFFSET]], %[[B_SIZE]] : i64
// CHECK: %[[B_FAST:.*]] = llvm.getelementptr %[[B_BASE]][%[[B_BYTES]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_f80(%[[A_FAST]])
// CHECK: llvm.call @use_i32(%[[B_FAST]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[A_FALLBACK:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: %[[B_FALLBACK:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_f80(%[[A_FALLBACK]])
// CHECK: llvm.call @use_i32(%[[B_FALLBACK]])

// CHECK-LABEL: llvm.func @exact_width_unsigned_lower(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr
// CHECK-SAME: %[[LOWER:[^:]+]]: i64
// CHECK-SAME: %[[COORDINATE:[^:]+]]: i64
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[FAST_INDEX:.*]] = llvm.sub %[[COORDINATE]], %[[FAST_ONE]] : i64
// CHECK: %[[FAST_LOWER:.*]] = llvm.sub %[[LOWER]], %[[FAST_ONE]] : i64
// CHECK: llvm.add %[[FAST_INDEX]], %[[FAST_LOWER]] : i64
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.sub %[[LOWER]], %[[FALLBACK_ONE]]
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32(%[[FALLBACK_ADDR]])

// CHECK-LABEL: llvm.func @negative_section_lower(
// CHECK: %[[NEGATIVE:.*]] = llvm.mlir.constant(-3 : i64) : i64
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[FAST_LOWER:.*]] = llvm.sub %[[NEGATIVE]], %[[FAST_ONE]] : i64
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.sub %[[NEGATIVE]], %[[FALLBACK_ONE]]
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32(%[[FALLBACK_ADDR]])

// CHECK-LABEL: llvm.func @kind_mapped_one_bit_widening(
// CHECK-SAME: %[[VALUE:[^:]+]]: i32
// CHECK: %[[BIT:.*]] = llvm.trunc %[[VALUE]] : i32 to i1
// CHECK: %[[WIDE:.*]] = llvm.sext %[[BIT]] : i1 to i64
// CHECK: llvm.return %[[WIDE]] : i64

// -----

// FIR-to-LLVM uses i64 for abstract index on every non-p32 layout. Verify that
// the byte-address path follows that contract even for a narrow pointer size.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:16:16"
} {
  func.func private @use_i32_p16(!fir.ref<i32>)

  func.func @pointer16(%a: !fir.box<!fir.array<?xi32>>,
                       %sectionLower: i64, %coordinate: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i64)
          -> !fir.ref<i32>
      fir.call @use_i32_p16(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
}

// CHECK-LABEL: llvm.func @pointer16(
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_p16(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_p16(%[[FALLBACK_ADDR]])

// -----

// A pointer wider than 64 bits uses the same FIR-to-LLVM i64 index contract.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:128:128"
} {
  func.func private @use_i32_p128(!fir.ref<i32>)
  func.func private @use_i32_p128_wide(!fir.ref<i32>)

  func.func @pointer128(%a: !fir.box<!fir.array<?xi32>>,
                        %sectionLower: i64, %coordinate: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i64)
          -> !fir.ref<i32>
      fir.call @use_i32_p128(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }

  // A non-step i128 operand exceeds the i64 FIR-to-LLVM address-index contract
  // even when the target pointer itself is wider, so it must remain generic.
  func.func @pointer128_wide_rejected(
      %a: !fir.box<!fir.array<?xi32>>, %coordinate: i128) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %c1, %c8, %c1
      : (index, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i128)
          -> !fir.ref<i32>
      fir.call @use_i32_p128_wide(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
}

// CHECK-LABEL: llvm.func @pointer128(
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_p128(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_p128(%[[FALLBACK_ADDR]])

// CHECK-LABEL: llvm.func @pointer128_wide_rejected(
// CHECK-NOT: llvm.icmp "eq"
// CHECK: llvm.getelementptr
// CHECK-NOT: llvm.icmp "eq"
// CHECK: llvm.call @use_i32_p128_wide
// CHECK-NOT: llvm.icmp "eq"
// CHECK: llvm.return

// FIR-LABEL: func.func @pointer128_wide_rejected(
// FIR-SAME: %[[A:[^:]+]]: !fir.box
// FIR-NOT: fir.box_dims %[[A]],
// FIR-NOT: fir.if
// FIR: fir.do_loop
// FIR: %[[ADDRESS:.*]] = fir.array_coor %[[A]]
// FIR: fir.call @use_i32_p128_wide(%[[ADDRESS]])
// FIR-NOT: fir.if
// FIR: return

// -----

// A p64 layout may use a narrower native GEP index field. FIR-to-LLVM still
// lowers abstract index arithmetic to i64 because the pointer itself is p64.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:64:64:64:32"
} {
  func.func private @use_i32_gep32(!fir.ref<i32>)

  func.func @pointer64_gep32(%a: !fir.box<!fir.array<?xi32>>,
                             %sectionLower: i64, %coordinate: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %sectionLower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %coordinate
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, i64)
          -> !fir.ref<i32>
      fir.call @use_i32_gep32(%address) : (!fir.ref<i32>) -> ()
    }
    return
  }
}

// CHECK-LABEL: llvm.func @pointer64_gep32(
// CHECK: llvm.cond_br %{{.*}}, ^[[FAST:bb[0-9]+]], ^[[FALLBACK:bb[0-9]+]]
// CHECK: ^[[FAST]]:
// CHECK: %[[FAST_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_gep32(%[[FAST_ADDR]])
// CHECK: ^[[FALLBACK]]:
// CHECK: %[[FALLBACK_ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.call @use_i32_gep32(%[[FALLBACK_ADDR]])
