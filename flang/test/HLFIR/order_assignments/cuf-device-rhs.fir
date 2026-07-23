// Test code generation of hlfir.region_assign with CUDA device RHS variables.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @host_rhs_still_saved(%lhs_ref: !fir.ref<!fir.array<10xf32>>,
                                %rhs_ref: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %lhs:2 = hlfir.declare %lhs_ref(%shape) {uniq_name = "lhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %rhs:2 = hlfir.declare %rhs_ref(%shape) {uniq_name = "rhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    hlfir.yield %rhs#0 : !fir.ref<!fir.array<10xf32>>
  } to {
    hlfir.yield %lhs#0 : !fir.ref<!fir.array<10xf32>>
  } user_defined_assign (%arg0: !fir.ref<!fir.array<10xf32>>) to (%arg1: !fir.ref<!fir.array<10xf32>>) {
    fir.call @assign_array(%arg1, %arg0) : (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @host_rhs_still_saved(
// CHECK:           %[[RHS:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "rhs"}
// CHECK:           %[[EXPR:.*]] = hlfir.as_expr %[[RHS]]#0 : (!fir.ref<!fir.array<10xf32>>) -> !hlfir.expr<10xf32>
// CHECK:           %[[TMP:.*]]:3 = hlfir.associate %[[EXPR]](%{{.*}}) {uniq_name = ".tmp.assign"}
// CHECK:           fir.call @assign_array(%{{.*}}, %[[TMP]]#0)

func.func @device_rhs_not_copied(%lhs_ref: !fir.ref<!fir.array<10xf32>>,
                                 %rhs_ref: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %lhs:2 = hlfir.declare %lhs_ref(%shape) {uniq_name = "lhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %rhs:2 = hlfir.declare %rhs_ref(%shape) {data_attr = #cuf.cuda<device>, uniq_name = "rhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    hlfir.yield %rhs#0 : !fir.ref<!fir.array<10xf32>>
  } to {
    hlfir.yield %lhs#0 : !fir.ref<!fir.array<10xf32>>
  } user_defined_assign (%arg0: !fir.ref<!fir.array<10xf32>>) to (%arg1: !fir.ref<!fir.array<10xf32>>) {
    fir.call @assign_array(%arg1, %arg0) : (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @device_rhs_not_copied(
// CHECK:           %[[LHS:.*]]:2 = hlfir.declare
// CHECK:           %[[RHS:.*]]:2 = hlfir.declare {{.*}} {data_attr = #cuf.cuda<device>
// CHECK-NOT:       hlfir.as_expr
// CHECK-NOT:       uniq_name = ".tmp.assign"
// CHECK:           fir.call @assign_array(%[[LHS]]#0, %[[RHS]]#0)

func.func @device_rhs_section_not_copied(%lhs_ref: !fir.ref<!fir.array<9xf32>>,
                                         %rhs_ref: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %lhs_shape = fir.shape %c9 : (index) -> !fir.shape<1>
  %rhs_shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %lhs:2 = hlfir.declare %lhs_ref(%lhs_shape) {uniq_name = "lhs"} : (!fir.ref<!fir.array<9xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<9xf32>>, !fir.ref<!fir.array<9xf32>>)
  %rhs:2 = hlfir.declare %rhs_ref(%rhs_shape) {data_attr = #cuf.cuda<device>, uniq_name = "rhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    %section = hlfir.designate %rhs#0 (%c2:%c10:%c1)  shape %lhs_shape : (!fir.ref<!fir.array<10xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<9xf32>>
    hlfir.yield %section : !fir.ref<!fir.array<9xf32>>
  } to {
    hlfir.yield %lhs#0 : !fir.ref<!fir.array<9xf32>>
  } user_defined_assign (%arg0: !fir.ref<!fir.array<9xf32>>) to (%arg1: !fir.ref<!fir.array<9xf32>>) {
    fir.call @assign_array9(%arg1, %arg0) : (!fir.ref<!fir.array<9xf32>>, !fir.ref<!fir.array<9xf32>>) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @device_rhs_section_not_copied(
// CHECK:           %[[LHS:.*]]:2 = hlfir.declare
// CHECK:           %[[RHS:.*]]:2 = hlfir.declare {{.*}} {data_attr = #cuf.cuda<device>
// CHECK:           %[[SECTION:.*]] = hlfir.designate %[[RHS]]#0
// CHECK-NOT:       hlfir.as_expr
// CHECK-NOT:       uniq_name = ".tmp.assign"
// CHECK:           fir.call @assign_array9(%[[LHS]]#0, %[[SECTION]])

func.func @parenthesized_device_rhs_still_saved(%lhs_ref: !fir.ref<!fir.array<10xf32>>,
                                                %rhs_ref: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %lhs:2 = hlfir.declare %lhs_ref(%shape) {uniq_name = "lhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %rhs:2 = hlfir.declare %rhs_ref(%shape) {data_attr = #cuf.cuda<device>, uniq_name = "rhs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    %expr = hlfir.as_expr %rhs#0 : (!fir.ref<!fir.array<10xf32>>) -> !hlfir.expr<10xf32>
    hlfir.yield %expr : !hlfir.expr<10xf32>
  } to {
    hlfir.yield %lhs#0 : !fir.ref<!fir.array<10xf32>>
  } user_defined_assign (%arg0: !hlfir.expr<10xf32>) to (%arg1: !fir.ref<!fir.array<10xf32>>) {
    %copy:3 = hlfir.associate %arg0(%shape) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>, i1)
    fir.call @assign_array(%arg1, %copy#0) : (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) -> ()
    hlfir.end_associate %copy#1, %copy#2 : !fir.ref<!fir.array<10xf32>>, i1
  }
  return
}

// CHECK-LABEL:   func.func @parenthesized_device_rhs_still_saved(
// CHECK:           %[[RHS:.*]]:2 = hlfir.declare {{.*}} {data_attr = #cuf.cuda<device>
// CHECK:           %[[EXPR:.*]] = hlfir.as_expr %[[RHS]]#0 : (!fir.ref<!fir.array<10xf32>>) -> !hlfir.expr<10xf32>
// CHECK:           %[[TMP:.*]]:3 = hlfir.associate %[[EXPR]](%{{.*}}) {uniq_name = ".tmp.assign"}
// CHECK:           %[[COPY_EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 : (!fir.ref<!fir.array<10xf32>>) -> !hlfir.expr<10xf32>
// CHECK:           hlfir.associate %[[COPY_EXPR]](%{{.*}}) {uniq_name = "adapt.valuebyref"}

func.func private @assign_array(!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) -> ()
func.func private @assign_array9(!fir.ref<!fir.array<9xf32>>, !fir.ref<!fir.array<9xf32>>) -> ()
