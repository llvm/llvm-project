// RUN: inter-opt %s --inter-coalesce-tuples | FileCheck %s

module {
  func.func @factor_common_fields(
      %base: !xemachine.reg<16, -1>,
      %first_dynamic: !xemachine.reg<1, -1>,
      %second_dynamic: !xemachine.reg<1, -1>) {
    %common0 = xemachine.imm 7 : i32
    %common1 = xemachine.imm 9 : i32
    %first0 = xemachine.mov %common0 {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first1 = xemachine.mov %common1 {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first = xemachine.update_tuple
        %base, %first0, %first_dynamic, %first1 {offsets = [2, 5, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %second0 = xemachine.mov %common0 {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second1 = xemachine.mov %common1 {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second = xemachine.update_tuple
        %base, %second0, %second_dynamic, %second1 {offsets = [2, 5, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %joined = xemachine.tuple_from_elements %first, %second
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @share_identical(
      %base: !xemachine.reg<16, -1>, %dynamic: !xemachine.reg<1, -1>) {
    %zero = xemachine.imm 0 : i32
    %first_common = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first = xemachine.update_tuple
        %base, %first_common, %dynamic {offsets = [2, 5]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %second_common = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second = xemachine.update_tuple
        %base, %second_common, %dynamic {offsets = [2, 5]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %joined = xemachine.tuple_from_elements %first, %second
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }
}

// CHECK-LABEL: func.func @factor_common_fields
// CHECK: [[COMMON0:%.*]] = xemachine.mov
// CHECK: [[COMMON1:%.*]] = xemachine.mov
// CHECK: [[TEMPLATE:%.*]] = xemachine.update_tuple %arg0, [[COMMON0]], [[COMMON1]] {offsets = [2, 7]}
// CHECK: [[FIRST:%.*]] = xemachine.update_tuple [[TEMPLATE]], %arg1 {offsets = [5]}
// CHECK: [[SECOND:%.*]] = xemachine.update_tuple [[TEMPLATE]], %arg2 {offsets = [5]}
// CHECK: xemachine.tuple_from_elements [[FIRST]], [[SECOND]]

// CHECK-LABEL: func.func @share_identical
// CHECK: [[COMMON:%.*]] = xemachine.mov
// CHECK: [[SHARED:%.*]] = xemachine.update_tuple %arg0, [[COMMON]], %arg1 {offsets = [2, 5]}
// CHECK-NOT: xemachine.update_tuple
// CHECK: xemachine.tuple_from_elements [[SHARED]], [[SHARED]]
