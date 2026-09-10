// RUN: inter-opt %s --inter-prepare-regalloc | FileCheck %s --check-prefix=PREP
// RUN: inter-opt %s --inter-prepare-regalloc > %t.once
// RUN: inter-opt %s --inter-prepare-regalloc --inter-prepare-regalloc > %t.twice
// RUN: diff %t.once %t.twice
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' | FileCheck %s --check-prefix=ALLOC

module {
  func.func @live_update_base() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %later = xemachine.add %base, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<32, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @materialized_base_owns_storage() attributes {
      xemachine.grf_count = 8 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %scalar = xemachine.mov %one {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i64) -> !xemachine.reg<2, -1>
    %base = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %replacement = xemachine.mov %one {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %last_scalar_use = xemachine.add %scalar, %scalar {execSize = 1 : i32,
        noMask} : (!xemachine.reg<2, -1>, !xemachine.reg<2, -1>, i64)
        -> !xemachine.reg<2, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [5]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %later = xemachine.add %base, %one {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    return
  }

  func.func @tuple_update_value_copy() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %a = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %b = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %tuple = xemachine.tuple_from_elements %a, %b
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %updated = xemachine.update_tuple %base, %tuple {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<32, -1>
    %later = xemachine.add %a, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @wide_live_update_base() attributes {
      xemachine.grf_count = 24 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %low = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %high = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %base = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<64, -1>
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<64, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<64, -1>
    %later = xemachine.add %base, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<64, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @repeated_update_base() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    xemachine.uniform_loop () {
      %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
          : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
          -> !xemachine.reg<32, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    return
  }

  func.func @duplicate_tuple_element() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %value = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %tuple = xemachine.tuple_from_elements %value, %value
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @incompatible_tuple_slots() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %a = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %b = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %first = xemachine.tuple_from_elements %a, %b
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %second = xemachine.tuple_from_elements %b, %a
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @fixed_element_virtual_tuple() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %value = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %tuple = xemachine.tuple_from_elements %r0, %value
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @external_branch_yield() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %result = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      xemachine.yield %external : !xemachine.reg<16, -1>
    } otherwise {
      %local = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.yield %local : !xemachine.reg<16, -1>
    } -> !xemachine.reg<16, -1>
    %later = xemachine.add %external, %result {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @dead_external_uniform_branch_yield() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %result = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      xemachine.yield %external : !xemachine.reg<16, -1>
    } otherwise {
      %local = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.yield %local : !xemachine.reg<16, -1>
    } -> !xemachine.reg<16, -1>
    return
  }

  func.func @uniform_branch_passthrough() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %result = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      xemachine.yield %external : !xemachine.reg<16, -1>
    } otherwise {
      xemachine.yield %external : !xemachine.reg<16, -1>
    } -> !xemachine.reg<16, -1>
    return
  }

  func.func @duplicate_uniform_branch_result() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %result:2 = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      xemachine.yield %external, %external
          : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
    } otherwise {
      %lhs = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %rhs = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.yield %lhs, %rhs
          : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
    } -> !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
    return
  }

  func.func @duplicate_live_loop_inits() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %loop:2 = xemachine.uniform_loop (%init, %init) {
    ^bb0(%lhs: !xemachine.reg<16, -1>, %rhs: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%lhs, %rhs : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %later = xemachine.add %init, %loop#0 {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @loop_backedge_swap() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %lhs_init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %rhs_init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %loop:2 = xemachine.uniform_loop (%lhs_init, %rhs_init) {
    ^bb0(%lhs: !xemachine.reg<16, -1>, %rhs: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%rhs, %lhs : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    return
  }

  func.func @loop_argument_used_after_next() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %loop = xemachine.uniform_loop (%init) {
    ^bb0(%iter: !xemachine.reg<16, -1>):
      %next = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %late = xemachine.add %iter, %one {execSize = 16 : i32, noMask}
          : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
          -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%next : !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    return
  }

  func.func @overlapping_loop_inits() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %low = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %high = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %whole = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %whole
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %loop:2 = xemachine.uniform_loop (%whole, %parts#0) {
    ^bb0(%wide: !xemachine.reg<32, -1>, %narrow: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%wide, %narrow : !xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
    } : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
    return
  }

  func.func @early_update_source() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @wrong_offset_update_alias() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %base
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %updated = xemachine.update_tuple %base, %parts#0 {offsets = [16]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @live_tuple_view_after_update() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %base
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %later = xemachine.add %parts#1, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @external_exec_if_view() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %result = xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
      %parts:2 = xemachine.tuple_to_elements %external
          : (!xemachine.reg<32, -1>)
          -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
      xemachine.yield %parts#0 : !xemachine.reg<16, -1>
    } otherwise {
      %local = xemachine.mov %one {execSize = 16 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.yield %local : !xemachine.reg<16, -1>
    } -> !xemachine.reg<16, -1>
    %later = xemachine.add %result, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @loop_init_alias_live_through() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %whole = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %whole
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %loop = xemachine.uniform_loop (%parts#0) {
    ^bb0(%iter: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%iter : !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    %later = xemachine.add %whole, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<32, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @loop_argument_view_used_after_next() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %init = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %loop = xemachine.uniform_loop (%init) {
    ^bb0(%iter: !xemachine.reg<32, -1>):
      %parts:2 = xemachine.tuple_to_elements %iter
          : (!xemachine.reg<32, -1>)
          -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
      %next = xemachine.mov %one {execSize = 32 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
      %late = xemachine.add %parts#0, %one {execSize = 16 : i32, noMask}
          : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
          -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%next : !xemachine.reg<32, -1>)
    } : (!xemachine.reg<32, -1>) -> !xemachine.reg<32, -1>
    return
  }

  func.func @simd16_copy_tail() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %low = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %high = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %base = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<48, -1>
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<48, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<48, -1>
    %later = xemachine.add %base, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<48, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @duplicate_update_source() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement, %replacement
        {offsets = [0, 16]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>,
           !xemachine.reg<16, -1>) -> !xemachine.reg<32, -1>
    return
  }

  func.func @intervening_base_view_read() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %base
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %replacement = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %read = xemachine.add %parts#1, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [16]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @fixed_shifted_tuple_view() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %source = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %source
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %view = xemachine.tuple_from_elements %parts#1
        : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, 0>
    return
  }

  func.func @subgrf_branch_yield() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 8 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<8, -1>
    %result = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      xemachine.yield %external : !xemachine.reg<8, -1>
    } otherwise {
      %local = xemachine.mov %one {execSize = 8 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<8, -1>
      xemachine.yield %local : !xemachine.reg<8, -1>
    } -> !xemachine.reg<8, -1>
    return
  }

  func.func @branch_backedge_swap() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %lhs_init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %rhs_init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %loop:2 = xemachine.uniform_loop (%lhs_init, %rhs_init) {
    ^bb0(%lhs: !xemachine.reg<16, -1>, %rhs: !xemachine.reg<16, -1>):
      %next:2 = xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
        xemachine.yield %rhs, %lhs
            : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
      } otherwise {
        xemachine.yield %lhs, %rhs
            : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
      } -> !xemachine.reg<16, -1>, !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%next#0, %next#1 : !xemachine.reg<16, -1>,
           !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    return
  }

  func.func @fixed_shifted_tuple_split() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %source = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %source
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, 0>)
    %later = xemachine.add %parts#1, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @fixed_upper_tuple_view() attributes {
      xemachine.grf_count = 3 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %source = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %source
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %view = xemachine.tuple_from_elements %parts#0
        : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, 2>
    return
  }

  func.func @sequential_loop_alias_inits() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %init = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %first = xemachine.uniform_loop (%init) {
    ^bb0(%iter: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%iter : !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    %second:2 = xemachine.uniform_loop (%first, %init) {
    ^bb0(%lhs: !xemachine.reg<16, -1>, %rhs: !xemachine.reg<16, -1>):
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%lhs, %rhs : !xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    return
  }

  func.func @aligned_tuple_view() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %source = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %source
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
    %view = xemachine.tuple_from_elements %parts#1
        : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    return
  }

  func.func @nested_repetitive_loop_init() attributes {
      xemachine.grf_count = 16 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %external = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    xemachine.uniform_loop () {
      %inner = xemachine.uniform_loop (%external) {
      ^bb0(%iter: !xemachine.reg<16, -1>):
        xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
            (%iter : !xemachine.reg<16, -1>)
      } : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    return
  }

  func.func @inconsistent_fixed_tuple_split() attributes {
      xemachine.grf_count = 4 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %one = xemachine.imm 1 : i32
    %source = xemachine.mov %one {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %parts:2 = xemachine.tuple_to_elements %source
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<16, 0>, !xemachine.reg<16, 2>)
    %later = xemachine.add %parts#1, %one {execSize = 16 : i32, noMask}
        : (!xemachine.reg<16, 2>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }
}

// PREP-LABEL: func.func @live_update_base
// PREP: [[BASE:%.*]] = xemachine.mov
// PREP: [[BASE_COPY:%.*]] = xemachine.mov [[BASE]] {{.*}}xemachine.regalloc_copy = "update-base"
// PREP-NEXT: [[UPDATE_COPY:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: [[UPDATED:%.*]] = xemachine.update_tuple [[BASE_COPY]], [[UPDATE_COPY]]
// PREP: xemachine.add [[BASE]],

// PREP-LABEL: func.func @materialized_base_owns_storage
// PREP: [[OWNED_BASE:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, -1>
// PREP: [[OWNED_BASE_COPY:%.*]] = xemachine.mov [[OWNED_BASE]] {{.*}}xemachine.regalloc_copy = "update-base"
// PREP: xemachine.update_tuple [[OWNED_BASE_COPY]],

// PREP-LABEL: func.func @tuple_update_value_copy
// PREP: [[TUPLE_UPDATE:%.*]] = xemachine.tuple_from_elements
// PREP: [[TUPLE_UPDATE_COPY:%.*]] = xemachine.mov [[TUPLE_UPDATE]] {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple {{%.*}}, [[TUPLE_UPDATE_COPY]]

// PREP-LABEL: func.func @wide_live_update_base
// PREP: [[WIDE_BASE:%.*]] = xemachine.tuple_from_elements
// PREP: [[WIDE_COPY0:%.*]] = xemachine.mov [[WIDE_BASE]] {{.*}}xemachine.regalloc_copy = "update-base"
// PREP-NEXT: [[WIDE_COPY1:%.*]] = xemachine.mov [[WIDE_BASE]] {{.*}}src0Sub = 32{{.*}}xemachine.regalloc_copy = "update-base"
// PREP-NEXT: [[WIDE_COPY:%.*]] = xemachine.tuple_from_elements [[WIDE_COPY0]], [[WIDE_COPY1]] {{.*}}xemachine.regalloc_copy = "update-base"
// PREP: xemachine.update_tuple [[WIDE_COPY]],

// PREP-LABEL: func.func @repeated_update_base
// PREP: xemachine.uniform_loop
// PREP: [[REPEATED_BASE_COPY:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-base"
// PREP-NEXT: [[REPEATED_UPDATE_COPY:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple [[REPEATED_BASE_COPY]], [[REPEATED_UPDATE_COPY]]

// PREP-LABEL: func.func @duplicate_tuple_element
// PREP: [[DUP:%.*]] = xemachine.mov
// PREP-NEXT: [[DUP_COPY:%.*]] = xemachine.mov [[DUP]] {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP-NEXT: xemachine.tuple_from_elements [[DUP]], [[DUP_COPY]]

// PREP-LABEL: func.func @incompatible_tuple_slots
// PREP: [[A:%.*]] = xemachine.mov
// PREP: [[B:%.*]] = xemachine.mov
// PREP: xemachine.tuple_from_elements [[A]], [[B]]
// PREP: [[B_COPY:%.*]] = xemachine.mov [[B]] {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP-NEXT: [[A_COPY:%.*]] = xemachine.mov [[A]] {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP-NEXT: xemachine.tuple_from_elements [[B_COPY]], [[A_COPY]]

// PREP-LABEL: func.func @fixed_element_virtual_tuple
// PREP: [[R0:%.*]] = xemachine.archreg 0
// PREP: [[FIXED_COPY:%.*]] = xemachine.mov [[R0]] {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP: xemachine.tuple_from_elements [[FIXED_COPY]],

// PREP-LABEL: func.func @external_branch_yield
// PREP: xemachine.uniform_if
// PREP: [[YIELD_COPY:%.*]] = xemachine.mov {{.*}}noMask{{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP-NEXT: xemachine.yield [[YIELD_COPY]]

// PREP-LABEL: func.func @dead_external_uniform_branch_yield
// PREP: xemachine.uniform_if
// PREP-NOT: xemachine.regalloc_copy = "branch-yield"
// PREP: return

// PREP-LABEL: func.func @uniform_branch_passthrough
// PREP: xemachine.uniform_if
// PREP-NOT: xemachine.regalloc_copy = "branch-yield"
// PREP: return

// PREP-LABEL: func.func @duplicate_uniform_branch_result
// PREP: xemachine.uniform_if
// PREP: [[DUPLICATE_RESULT_COPY0:%.*]] = xemachine.mov {{.*}}noMask{{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP-NEXT: [[DUPLICATE_RESULT_COPY1:%.*]] = xemachine.mov {{.*}}noMask{{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP-NEXT: xemachine.yield [[DUPLICATE_RESULT_COPY0]], [[DUPLICATE_RESULT_COPY1]]

// PREP-LABEL: func.func @duplicate_live_loop_inits
// PREP: [[INIT:%.*]] = xemachine.mov
// PREP-NEXT: [[INIT_COPY0:%.*]] = xemachine.mov [[INIT]] {{.*}}noMask{{.*}}xemachine.regalloc_copy = "loop-init"
// PREP-NEXT: [[INIT_COPY1:%.*]] = xemachine.mov [[INIT]] {{.*}}noMask{{.*}}xemachine.regalloc_copy = "loop-init"
// PREP-NEXT: {{%.*}}:2 = xemachine.uniform_loop([[INIT_COPY0]], [[INIT_COPY1]])

// PREP-LABEL: func.func @loop_backedge_swap
// PREP: xemachine.uniform_loop
// PREP: [[SNAP0:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "loop-snapshot"
// PREP-NEXT: [[SNAP1:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "loop-snapshot"
// PREP-NEXT: [[DEST0:%.*]] = xemachine.mov [[SNAP0]] {{.*}}xemachine.regalloc_copy = "loop-backedge"
// PREP-NEXT: [[DEST1:%.*]] = xemachine.mov [[SNAP1]] {{.*}}xemachine.regalloc_copy = "loop-backedge"
// PREP-NEXT: xemachine.continue_if {{.*}}([[DEST0]], [[DEST1]]

// PREP-LABEL: func.func @loop_argument_used_after_next
// PREP: [[LATE_SNAPSHOT:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "loop-snapshot"
// PREP-NEXT: [[LATE_DEST:%.*]] = xemachine.mov [[LATE_SNAPSHOT]] {{.*}}xemachine.regalloc_copy = "loop-backedge"
// PREP-NEXT: xemachine.continue_if {{.*}}([[LATE_DEST]]

// PREP-LABEL: func.func @overlapping_loop_inits
// PREP: [[WHOLE:%.*]] = xemachine.tuple_from_elements
// PREP: [[PARTS:%.*]]:2 = xemachine.tuple_to_elements [[WHOLE]]
// PREP: [[OVERLAP_COPY:%.*]] = xemachine.mov [[PARTS]]#0 {{.*}}xemachine.regalloc_copy = "loop-init"
// PREP-NEXT: {{%.*}}:2 = xemachine.uniform_loop([[WHOLE]], [[OVERLAP_COPY]])

// PREP-LABEL: func.func @early_update_source
// PREP: [[EARLY_UPDATE:%.*]] = xemachine.mov
// PREP: [[EARLY_BASE:%.*]] = xemachine.mov
// PREP-NEXT: [[EARLY_UPDATE_COPY:%.*]] = xemachine.mov [[EARLY_UPDATE]] {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple [[EARLY_BASE]], [[EARLY_UPDATE_COPY]]

// PREP-LABEL: func.func @wrong_offset_update_alias
// PREP: [[OFFSET_BASE:%.*]] = xemachine.mov
// PREP: [[OFFSET_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[OFFSET_BASE]]
// PREP-NEXT: [[OFFSET_UPDATE_COPY:%.*]] = xemachine.mov [[OFFSET_PARTS]]#0 {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple [[OFFSET_BASE]], [[OFFSET_UPDATE_COPY]]

// PREP-LABEL: func.func @live_tuple_view_after_update
// PREP: [[VIEW_BASE:%.*]] = xemachine.mov
// PREP: [[VIEW_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[VIEW_BASE]]
// PREP: [[VIEW_BASE_COPY:%.*]] = xemachine.mov [[VIEW_BASE]] {{.*}}xemachine.regalloc_copy = "update-base"
// PREP: xemachine.update_tuple [[VIEW_BASE_COPY]],
// PREP: xemachine.add [[VIEW_PARTS]]#1,

// PREP-LABEL: func.func @external_exec_if_view
// PREP: [[EXEC_EXTERNAL:%.*]] = xemachine.mov
// PREP: xemachine.exec_if
// PREP: [[EXEC_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[EXEC_EXTERNAL]]
// PREP-NEXT: [[EXEC_COPY:%.*]] = xemachine.mov [[EXEC_PARTS]]#0 {xemachine.regalloc_copy = "branch-yield"}
// PREP-NEXT: xemachine.yield [[EXEC_COPY]]
// PREP: } otherwise {
// PREP-NEXT: [[EXEC_LOCAL:%.*]] = xemachine.mov {{.*}}noMask
// PREP-NEXT: xemachine.yield [[EXEC_LOCAL]]

// PREP-LABEL: func.func @loop_init_alias_live_through
// PREP: [[LIVE_WHOLE:%.*]] = xemachine.mov
// PREP: [[LIVE_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[LIVE_WHOLE]]
// PREP-NEXT: [[LIVE_INIT_COPY:%.*]] = xemachine.mov [[LIVE_PARTS]]#0 {{.*}}xemachine.regalloc_copy = "loop-init"
// PREP-NEXT: {{%.*}} = xemachine.uniform_loop([[LIVE_INIT_COPY]])
// PREP: xemachine.add [[LIVE_WHOLE]],

// PREP-LABEL: func.func @loop_argument_view_used_after_next
// PREP: xemachine.uniform_loop
// PREP: [[LOOP_VIEW:%.*]]:2 = xemachine.tuple_to_elements
// PREP: [[VIEW_SNAPSHOT:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "loop-snapshot"
// PREP-NEXT: [[VIEW_DEST:%.*]] = xemachine.mov [[VIEW_SNAPSHOT]] {{.*}}xemachine.regalloc_copy = "loop-backedge"
// PREP-NEXT: xemachine.continue_if {{.*}}([[VIEW_DEST]]

// PREP-LABEL: func.func @simd16_copy_tail
// PREP: [[TAIL_BASE:%.*]] = xemachine.tuple_from_elements
// PREP: [[TAIL_COPY0:%.*]] = xemachine.mov [[TAIL_BASE]] {{.*}}execSize = 32{{.*}}xemachine.regalloc_copy = "update-base"
// PREP-NEXT: [[TAIL_COPY1:%.*]] = xemachine.mov [[TAIL_BASE]] {{.*}}src0Sub = 32{{.*}}xemachine.regalloc_copy = "update-base"{{.*}}-> !xemachine.reg<16, -1>
// PREP-NEXT: xemachine.tuple_from_elements [[TAIL_COPY0]], [[TAIL_COPY1]] {{.*}}xemachine.regalloc_copy = "update-base"

// PREP-LABEL: func.func @duplicate_update_source
// PREP: [[DUP_UPDATE_BASE:%.*]] = xemachine.mov {{.*}}execSize = 32
// PREP-NEXT: [[DUP_UPDATE:%.*]] = xemachine.mov
// PREP: [[DUP_UPDATE_COPY0:%.*]] = xemachine.mov [[DUP_UPDATE]] {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple {{%.*}}, [[DUP_UPDATE_COPY0]], [[DUP_UPDATE]]

// PREP-LABEL: func.func @intervening_base_view_read
// PREP: [[READ_BASE:%.*]] = xemachine.mov
// PREP: [[READ_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[READ_BASE]]
// PREP: [[READ_REPLACEMENT:%.*]] = xemachine.mov
// PREP-NEXT: xemachine.add [[READ_PARTS]]#1,
// PREP-NEXT: [[READ_COPY:%.*]] = xemachine.mov [[READ_REPLACEMENT]] {{.*}}xemachine.regalloc_copy = "update-value"
// PREP-NEXT: xemachine.update_tuple [[READ_BASE]], [[READ_COPY]]

// PREP-LABEL: func.func @fixed_shifted_tuple_view
// PREP: [[SHIFT_PARTS:%.*]]:2 = xemachine.tuple_to_elements
// PREP-NEXT: [[SHIFT_COPY:%.*]] = xemachine.mov [[SHIFT_PARTS]]#1 {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP-NEXT: xemachine.tuple_from_elements [[SHIFT_COPY]] {{.*}}!xemachine.reg<16, 0>

// PREP-LABEL: func.func @subgrf_branch_yield
// PREP: xemachine.uniform_if
// PREP-NOT: xemachine.regalloc_copy = "branch-yield"
// PREP: return

// PREP-LABEL: func.func @branch_backedge_swap
// PREP: xemachine.uniform_if
// PREP: [[BRANCH_SNAP0:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "branch-snapshot"
// PREP-NEXT: [[BRANCH_SNAP1:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "branch-snapshot"
// PREP-NEXT: [[BRANCH_DEST0:%.*]] = xemachine.mov [[BRANCH_SNAP0]] {{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP-NEXT: [[BRANCH_DEST1:%.*]] = xemachine.mov [[BRANCH_SNAP1]] {{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP-NEXT: xemachine.yield [[BRANCH_DEST0]], [[BRANCH_DEST1]]

// PREP-LABEL: func.func @fixed_shifted_tuple_split
// PREP: [[SPLIT_SOURCE:%.*]] = xemachine.mov
// PREP-NEXT: [[SPLIT_PARTS:%.*]]:2 = xemachine.tuple_to_elements [[SPLIT_SOURCE]] {{.*}}!xemachine.reg<16, -1>, !xemachine.reg<16, -1>
// PREP-NEXT: [[SPLIT_COPY:%.*]] = xemachine.mov [[SPLIT_PARTS]]#1 {{.*}}xemachine.regalloc_copy = "tuple-element"{{.*}}!xemachine.reg<16, 0>

// PREP-LABEL: func.func @fixed_upper_tuple_view
// PREP: [[UPPER_PARTS:%.*]]:2 = xemachine.tuple_to_elements
// PREP-NEXT: [[UPPER_COPY:%.*]] = xemachine.mov [[UPPER_PARTS]]#0 {{.*}}xemachine.regalloc_copy = "tuple-element"
// PREP-NEXT: xemachine.tuple_from_elements [[UPPER_COPY]] {{.*}}!xemachine.reg<16, 2>

// PREP-LABEL: func.func @sequential_loop_alias_inits
// PREP: [[FIRST_LOOP:%.*]] = xemachine.uniform_loop
// PREP: {{%.*}}:2 = xemachine.uniform_loop([[FIRST_LOOP]],

// PREP-LABEL: func.func @aligned_tuple_view
// PREP: [[ALIGNED_PARTS:%.*]]:2 = xemachine.tuple_to_elements
// PREP-NEXT: xemachine.tuple_from_elements [[ALIGNED_PARTS]]#1

// PREP-LABEL: func.func @nested_repetitive_loop_init
// PREP: [[NESTED_EXTERNAL:%.*]] = xemachine.mov
// PREP: xemachine.uniform_loop
// PREP-NEXT: [[NESTED_COPY:%.*]] = xemachine.mov [[NESTED_EXTERNAL]] {{.*}}xemachine.regalloc_copy = "loop-init"
// PREP-NEXT: {{%.*}} = xemachine.uniform_loop([[NESTED_COPY]])

// PREP-LABEL: func.func @inconsistent_fixed_tuple_split
// PREP: [[CONFLICT_PARTS:%.*]]:2 = xemachine.tuple_to_elements {{.*}}!xemachine.reg<16, 0>, !xemachine.reg<16, -1>
// PREP-NEXT: [[CONFLICT_COPY:%.*]] = xemachine.mov [[CONFLICT_PARTS]]#1 {{.*}}xemachine.regalloc_copy = "tuple-element"{{.*}}!xemachine.reg<16, 2>

// ALLOC-LABEL: func.func @duplicate_tuple_element
// ALLOC: [[ALLOC_DUP:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 0>
// ALLOC-NEXT: [[ALLOC_DUP_COPY:%.*]] = xemachine.mov [[ALLOC_DUP]] {{.*}}-> !xemachine.reg<16, 1>
// ALLOC-NEXT: xemachine.tuple_from_elements [[ALLOC_DUP]], [[ALLOC_DUP_COPY]]

// ALLOC-LABEL: func.func @external_branch_yield
// ALLOC: [[ALLOC_EXTERNAL:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 0>
// ALLOC: [[ALLOC_RESULT:%.*]] = xemachine.uniform_if
// ALLOC: } -> !xemachine.reg<16, 1>
// ALLOC: xemachine.add [[ALLOC_EXTERNAL]], [[ALLOC_RESULT]]

// ALLOC-LABEL: func.func @dead_external_uniform_branch_yield
// ALLOC: [[DEAD_EXTERNAL:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, [[DEAD_SLOT:[0-9]+]]>
// ALLOC: xemachine.uniform_if
// ALLOC: xemachine.yield [[DEAD_EXTERNAL]] : !xemachine.reg<16, [[DEAD_SLOT]]>
// ALLOC: xemachine.yield {{%.*}} : !xemachine.reg<16, [[DEAD_SLOT]]>

// ALLOC-LABEL: func.func @uniform_branch_passthrough
// ALLOC: [[PASSTHROUGH:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, [[PASSTHROUGH_SLOT:[0-9]+]]>
// ALLOC: xemachine.uniform_if
// ALLOC: xemachine.yield [[PASSTHROUGH]] : !xemachine.reg<16, [[PASSTHROUGH_SLOT]]>
// ALLOC: xemachine.yield [[PASSTHROUGH]] : !xemachine.reg<16, [[PASSTHROUGH_SLOT]]>

// ALLOC-LABEL: func.func @duplicate_uniform_branch_result
// ALLOC: xemachine.uniform_if
// ALLOC: xemachine.yield {{%.*}}, {{%.*}} : !xemachine.reg<16, [[DUP_SLOT0:[0-9]+]]>, !xemachine.reg<16, [[DUP_SLOT1:[0-9]+]]>
// ALLOC-NOT: !xemachine.reg<16, [[DUP_SLOT0]]>, !xemachine.reg<16, [[DUP_SLOT0]]>

// ALLOC-LABEL: func.func @duplicate_live_loop_inits
// ALLOC: xemachine.uniform_loop
// ALLOC: ^{{.*}}({{%.*}}: !xemachine.reg<16, 1>, {{%.*}}: !xemachine.reg<16, 2>):

// ALLOC-LABEL: func.func @loop_backedge_swap
// ALLOC: xemachine.uniform_loop
// ALLOC: ^{{.*}}([[ALLOC_LHS:%.*]]: !xemachine.reg<16, 0>, [[ALLOC_RHS:%.*]]: !xemachine.reg<16, 1>):
// ALLOC: [[ALLOC_SNAP0:%.*]] = xemachine.mov [[ALLOC_RHS]] {{.*}}-> !xemachine.reg<16, 2>
// ALLOC-NEXT: [[ALLOC_SNAP1:%.*]] = xemachine.mov [[ALLOC_LHS]] {{.*}}-> !xemachine.reg<16, 3>
// ALLOC-NEXT: [[ALLOC_DEST0:%.*]] = xemachine.mov [[ALLOC_SNAP0]] {{.*}}-> !xemachine.reg<16, 0>
// ALLOC-NEXT: [[ALLOC_DEST1:%.*]] = xemachine.mov [[ALLOC_SNAP1]] {{.*}}-> !xemachine.reg<16, 1>
// ALLOC-LABEL: func.func @fixed_shifted_tuple_view
// ALLOC: [[ALLOC_SHIFT_COPY:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 0>
// ALLOC-NEXT: xemachine.tuple_from_elements [[ALLOC_SHIFT_COPY]] {{.*}}!xemachine.reg<16, 0>
// ALLOC-LABEL: func.func @fixed_shifted_tuple_split
// ALLOC: [[ALLOC_SPLIT_COPY:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 0>
// ALLOC-LABEL: func.func @fixed_upper_tuple_view
// ALLOC: [[ALLOC_UPPER_COPY:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 2>
// ALLOC-LABEL: func.func @sequential_loop_alias_inits
// ALLOC: [[SLOT_INIT:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "loop-init"{{.*}}-> !xemachine.reg<16, [[SLOT:[0-9]+]]>
// ALLOC: [[SLOT_RESULT:%.*]] = xemachine.uniform_loop([[SLOT_INIT]])
// ALLOC: ^{{.*}}([[SLOT_ARG:%.*]]: !xemachine.reg<16, [[SLOT]]>):
// ALLOC: xemachine.continue_if {{.*}}([[SLOT_ARG]] : !xemachine.reg<16, [[SLOT]]>)
// ALLOC: } : (!xemachine.reg<16, [[SLOT]]>) -> !xemachine.reg<16, [[SLOT]]>
// ALLOC-LABEL: func.func @inconsistent_fixed_tuple_split
// ALLOC: [[ALLOC_CONFLICT_COPY:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 2>
// ALLOC-NOT: !xemachine.reg<{{[0-9]+}}, -1>
