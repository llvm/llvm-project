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

  func.func @do_not_share_across_loop(
      %base: !xemachine.reg<16, -1>, %flag: !xemachine.arf<f, 2, 0>) {
    %zero = xemachine.imm 0 : i32
    %before_common0 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %before_common1 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %before = xemachine.update_tuple
        %base, %before_common0, %before_common1 {offsets = [3, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    xemachine.uniform_loop () {
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    %after_common0 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %after_common1 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %after = xemachine.update_tuple
        %base, %after_common0, %after_common1 {offsets = [3, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %joined = xemachine.tuple_from_elements %before, %after
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @prefer_destinationless_send_descriptor(
      %base: !xemachine.reg<16, -1>,
      %first_dynamic: !xemachine.reg<1, -1>,
      %second_dynamic: !xemachine.reg<1, -1>,
      %third_dynamic: !xemachine.reg<1, -1>) {
    %zero = xemachine.imm 0 : i32
    %first_common0 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first_common1 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first_common2 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %first = xemachine.update_tuple
        %base, %first_common0, %first_common1, %first_common2, %first_dynamic
        {offsets = [2, 3, 4, 5]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %first_dst, %first_token = xemachine.send ugm %first
        {desc = 1 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %second_common0 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second_common1 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second_common2 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %second = xemachine.update_tuple
        %base, %second_common0, %second_common1, %second_common2, %second_dynamic
        {offsets = [2, 3, 4, 6]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %second_dst, %second_token = xemachine.send ugm %second
        {desc = 2 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %third_common0 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %third_common1 = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %third = xemachine.update_tuple
        %base, %third_common0, %third_common1, %third_dynamic
        {offsets = [2, 3, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %third_dst, %third_token = xemachine.send ugm %third
        {desc = 1 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
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

// CHECK-LABEL: func.func @do_not_share_across_loop
// CHECK: [[BEFORE:%.*]] = xemachine.update_tuple
// CHECK: xemachine.uniform_loop
// CHECK: [[AFTER:%.*]] = xemachine.update_tuple
// CHECK: xemachine.tuple_from_elements [[BEFORE]], [[AFTER]]

// CHECK-LABEL: func.func @prefer_destinationless_send_descriptor
// CHECK: [[TEMPLATE:%.*]] = xemachine.update_tuple %arg0, {{.*}} {offsets = [2, 3]}
// CHECK: [[FIRST:%.*]] = xemachine.update_tuple [[TEMPLATE]]
// CHECK: xemachine.send ugm [[FIRST]] {{.*}}desc = 1
// CHECK: [[SECOND:%.*]] = xemachine.update_tuple %arg0
// CHECK: xemachine.send ugm [[SECOND]] {{.*}}desc = 2
// CHECK: [[THIRD:%.*]] = xemachine.update_tuple [[TEMPLATE]]
// CHECK: xemachine.send ugm [[THIRD]] {{.*}}desc = 1
