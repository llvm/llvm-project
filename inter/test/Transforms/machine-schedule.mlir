// RUN: inter-opt --inter-machine-schedule %s | FileCheck %s
// RUN: inter-opt --inter-machine-schedule --inter-machine-schedule %s | FileCheck %s
// RUN: inter-opt --inter-machine-schedule %s > %t.once
// RUN: inter-opt --inter-machine-schedule --inter-machine-schedule %s > %t.twice
// RUN: diff %t.once %t.twice

module {
  // CHECK-LABEL: func.func @payload_prologue
  // CHECK: xemachine.payload_prologue {
  // CHECK: [[BEFORE:%.*]] = xemachine.mov
  // CHECK: xemachine.payload_prologue_end
  // CHECK: }
  // CHECK: xemachine.sync allwr
  // CHECK: [[AFTER:%.*]] = xemachine.mov
  func.func @payload_prologue() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %one = xemachine.imm 1 : i32
    xemachine.payload_prologue {
      %before = xemachine.mov %one {execSize = 1 : i32, noMask}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.payload_prologue_end
    }
    %boundary = xemachine.sync allwr : !xemachine.mem.token
    %after = xemachine.mov %one {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @alu_gap_fill() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %filler = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @no_inst_closure() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %late = xemachine.imm 1 : i32
    %filler = xemachine.mov %late {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @newly_ready_no_inst_closure() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %filler = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %alias = xemachine.tuple_from_elements %filler
        : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    return
  }

  func.func @pipe_compatibility() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32)
        -> !xemachine.reg<16, -1>
    %integer_filler = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %floating_filler = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, f32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @pressure_guard() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %wide = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<64, -1>
    %dst, %token = xemachine.send ugm %wide
        {desc = 134217728 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<64, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    return
  }

  func.func @alias_aware_pressure_guard() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %live = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, 2>
    %consumer = xemachine.add %producer, %live {execSize = 16 : i32}
        : (!xemachine.reg<16, 2>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, 3>
    %candidate = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, f32) -> !xemachine.reg<16, -1>
    %base = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %replacement = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<16, -1>
    %dst, %token = xemachine.send ugm %updated
        {desc = 134217728 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    return
  }

  func.func @component_pressure_guard() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %low = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.arf<acc, 16, 0>
    %high = xemachine.mov %producer {execSize = 16 : i32}
        : (!xemachine.arf<acc, 16, 0>, f32) -> !xemachine.reg<16, -1>
    %tuple = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %dst, %token = xemachine.send ugm %tuple
        {desc = 134217728 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<32, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %candidate = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, f32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @fixed_component_pressure_guard() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r6 = xemachine.archreg 6 : !xemachine.reg<16, 6>
    %one = xemachine.imm 1 : i32
    %fixed = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 5>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.arf<acc, 16, 0>
    %consumer = xemachine.add %producer, %fixed {execSize = 16 : i32}
        : (!xemachine.arf<acc, 16, 0>, !xemachine.reg<16, 5>, f32)
        -> !xemachine.arf<acc, 16, 0>
    %candidate = xemachine.mov %r6 {execSize = 16 : i32}
        : (!xemachine.reg<16, 6>, f32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @cross_region_pressure_guard(
      %flag: !xemachine.arf<f, 2, 0>) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r5 = xemachine.archreg 5 : !xemachine.reg<16, 5>
    %pre = xemachine.mov %r5 {execSize = 16 : i32}
        : (!xemachine.reg<16, 5>, f32) -> !xemachine.arf<acc, 16, 0>
    xemachine.uniform_loop () {
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.arf<acc, 16, 0>
    %consumer = xemachine.add %producer, %r5 {execSize = 16 : i32}
        : (!xemachine.arf<acc, 16, 0>, !xemachine.reg<16, 5>, f32)
        -> !xemachine.arf<acc, 16, 0>
    %candidate = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, f32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @send_order_gap() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %first_dst, %first_token = xemachine.send ugm %r0
        {desc = 33554432 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %second_dst, %second_token = xemachine.send ugm %r0 dep %first_token
        {desc = 33554434 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %f0 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f1 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f2 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f3 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f4 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f5 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f6 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f7 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %f8 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @send_payload_pressure_guard() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %payload = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, -1>
    %first_dst, %first_token = xemachine.send ugm %payload
        {desc = 33554432 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %second_dst, %second_token = xemachine.send ugm %r0 dep %first_token
        {desc = 33554434 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %candidate = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, -1>
    return
  }

  func.func @unordered_memory() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %dependent_address = xemachine.add %producer, %producer
        {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %blocked_dst, %blocked_token = xemachine.send ugm %dependent_address
        {desc = 33554432 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %independent_dst, %independent_token = xemachine.send ugm %r0
        {desc = 33554434 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    return
  }

  func.func @token_ordered_memory() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %dependent_address = xemachine.add %producer, %producer
        {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %blocked_dst, %blocked_token = xemachine.send ugm %dependent_address
        {desc = 33554432 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %ordered_dst, %ordered_token = xemachine.send ugm %r0 dep %blocked_token
        {desc = 33554434 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    return
  }

  func.func @arf_war() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %a0 = xemachine.arfreg a0, 0 : !xemachine.arf<a0, 16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %a0 {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.arf<a0, 16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.and %r0, %one {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.arf<a0, 16, 0>
    return
  }

  func.func @arf_waw() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %writer0 = xemachine.mov %producer {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, -1>, f32) -> !xemachine.arf<a0, 16, 0>
    %writer1 = xemachine.and %r0, %one {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.arf<a0, 16, 0>
    return
  }

  func.func @arf_raw() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %a0 = xemachine.arfreg a0, 0 : !xemachine.arf<a0, 16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.mov %producer {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, -1>, f32) -> !xemachine.arf<a0, 16, 0>
    %reader = xemachine.and %a0, %one {execSize = 1 : i32, noMask}
        : (!xemachine.arf<a0, 16, 0>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @virtual_arf_files() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %a0 = xemachine.arfreg a0, -1 : !xemachine.arf<a0, 16, -1>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %a0 {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.arf<a0, 16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.and %r0, %one {execSize = 1 : i32, noMask}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.arf<f, 2, -1>
    return
  }

  func.func @fixed_grf_war() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r1 = xemachine.archreg 1 : !xemachine.reg<16, 1>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %r1 {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, 1>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 1>
    return
  }

  func.func @fixed_grf_waw() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %writer0 = xemachine.mov %producer {execSize = 8 : i32}
        : (!xemachine.reg<16, -1>, f32) -> !xemachine.reg<16, 1>
    %writer1 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 1>
    return
  }

  func.func @fixed_grf_raw() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r1 = xemachine.archreg 1 : !xemachine.reg<16, 1>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.mov %producer {execSize = 8 : i32}
        : (!xemachine.reg<16, -1>, f32) -> !xemachine.reg<16, 1>
    %reader = xemachine.and %r1, %one {execSize = 8 : i32}
        : (!xemachine.reg<16, 1>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }

  func.func @fixed_grf_tuple_alias_war() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r1 = xemachine.archreg 1 : !xemachine.reg<16, 1>
    %one = xemachine.imm 1 : i32
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %r1 {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, 1>, f32)
        -> !xemachine.reg<16, -1>
    %writer = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %fixed = xemachine.tuple_from_elements %writer
        : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, 1>
    return
  }

  func.func @destructive_tuple() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %base {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, f32)
        -> !xemachine.reg<16, -1>
    %replacement = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @destructive_tuple_alias_chain() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %base {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, f32)
        -> !xemachine.reg<16, -1>
    %low = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %high = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %replacement = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @destructive_tuple_across_region(
      %flag: !xemachine.arf<f, 2, 0>) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %base = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %reader = xemachine.add %producer, %base {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, f32)
        -> !xemachine.reg<16, -1>
    %low = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %high = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %replacement = xemachine.tuple_from_elements %low, %high
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    xemachine.uniform_loop () {
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    %updated = xemachine.update_tuple %base, %replacement {offsets = [0]}
        : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<32, -1>
    return
  }

  func.func @interleaved_loop_dpas_chains(
      %flag: !xemachine.arf<f, 2, 0>) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %a = xemachine.archreg 20 : !xemachine.reg<64, 20>
    %b = xemachine.archreg 24 : !xemachine.reg<128, 24>
    %zero = xemachine.imm 0 : f32
    %acc0 = xemachine.mov %zero {execSize = 16 : i32}
        : (!xemachine.imm, f32) -> !xemachine.reg<128, -1>
    %acc1 = xemachine.mov %zero {execSize = 16 : i32}
        : (!xemachine.imm, f32) -> !xemachine.reg<128, -1>
    %loop:2 = xemachine.uniform_loop (%acc0, %acc1) {
    ^bb0(%iter0: !xemachine.reg<128, -1>,
         %iter1: !xemachine.reg<128, -1>):
      %chain0_first = xemachine.dpas %a, %b, %iter0 {
          aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
          : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
             !xemachine.reg<128, -1>) -> !xemachine.reg<128, -1>
      %chain0_second = xemachine.dpas %a, %b, %chain0_first {
          aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
          : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
             !xemachine.reg<128, -1>) -> !xemachine.reg<128, -1>
      %chain1_first = xemachine.dpas %a, %b, %iter1 {
          aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
          : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
             !xemachine.reg<128, -1>) -> !xemachine.reg<128, -1>
      %chain1_second = xemachine.dpas %a, %b, %chain1_first {
          aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
          : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
             !xemachine.reg<128, -1>) -> !xemachine.reg<128, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%chain0_second, %chain1_second
           : !xemachine.reg<128, -1>, !xemachine.reg<128, -1>)
    } : (!xemachine.reg<128, -1>, !xemachine.reg<128, -1>)
        -> (!xemachine.reg<128, -1>, !xemachine.reg<128, -1>)
    return
  }

  func.func @loop_carry(%flag: !xemachine.arf<f, 2, 0>) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %init = xemachine.mov %one {execSize = 16 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %loop = xemachine.uniform_loop (%init) {
    ^bb0(%iter: !xemachine.reg<16, -1>):
      %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
          : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
          -> !xemachine.reg<16, -1>
      %reader = xemachine.add %producer, %iter {execSize = 16 : i32}
          : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
          -> !xemachine.reg<16, -1>
      %next = xemachine.mov %one {execSize = 16 : i32}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%next : !xemachine.reg<16, -1>)
    } : (!xemachine.reg<16, -1>) -> !xemachine.reg<16, -1>
    return
  }

  func.func @loop_alias_carry(%flag: !xemachine.arf<f, 2, 0>) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %init = xemachine.mov %one {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %loop = xemachine.uniform_loop (%init) {
    ^bb0(%iter: !xemachine.reg<32, -1>):
      %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
          : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
          -> !xemachine.reg<16, -1>
      %reader = xemachine.add %producer, %iter {execSize = 16 : i32}
          : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, f32)
          -> !xemachine.reg<16, -1>
      %next0 = xemachine.mov %one {execSize = 16 : i32}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %next1 = xemachine.mov %one {execSize = 16 : i32}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %next = xemachine.tuple_from_elements %next0, %next1
          : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
          -> !xemachine.reg<32, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
          (%next : !xemachine.reg<32, -1>)
    } : (!xemachine.reg<32, -1>) -> !xemachine.reg<32, -1>
    return
  }

  func.func @eot_boundary() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    xemachine.eot %r0 : !xemachine.reg<16, 0>
    return
  }

  func.func @raw_eot_boundary() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %terminal_dst, %terminal_token = xemachine.send ugm %r0
        {desc = 33554432 : i32, eot, exdesc = 0 : i32, noMask,
         sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    return
  }

  func.func @nested_region() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    %pre_producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %pre_consumer = xemachine.add %pre_producer, %pre_producer
        {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %pre_filler = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    xemachine.uniform_loop () {
      %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
          : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
          -> !xemachine.reg<16, -1>
      %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
          : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
          -> !xemachine.reg<16, -1>
      %filler = xemachine.mov %one {execSize = 8 : i32}
          : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    %post_producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %post_consumer = xemachine.add %post_producer, %post_producer
        {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    %post_filler = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// CHECK-LABEL: func.func @alu_gap_fill
// CHECK: [[ALU_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALU_FILLER:%.*]] = xemachine.mov
// CHECK-NEXT: [[ALU_CONSUMER:%.*]] = xemachine.add [[ALU_PRODUCER]], [[ALU_PRODUCER]]

// CHECK-LABEL: func.func @no_inst_closure
// CHECK: [[LATE_IMM:%.*]] = xemachine.imm 1
// CHECK-NEXT: [[NOINST_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[NOINST_FILLER:%.*]] = xemachine.mov [[LATE_IMM]]
// CHECK-NEXT: [[NOINST_CONSUMER:%.*]] = xemachine.add [[NOINST_PRODUCER]], [[NOINST_PRODUCER]]

// CHECK-LABEL: func.func @newly_ready_no_inst_closure
// CHECK: [[CLOSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[CLOSURE_FILLER:%.*]] = xemachine.mov
// CHECK-NEXT: [[CLOSURE_ALIAS:%.*]] = xemachine.tuple_from_elements [[CLOSURE_FILLER]]
// CHECK-NEXT: [[CLOSURE_CONSUMER:%.*]] = xemachine.add [[CLOSURE_PRODUCER]], [[CLOSURE_PRODUCER]]

// CHECK-LABEL: func.func @pipe_compatibility
// CHECK: [[PIPE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[FLOAT_FILLER:%.*]] = xemachine.mov {{.*}}f32
// CHECK-NEXT: [[INTEGER_FILLER:%.*]] = xemachine.mov {{.*}}i32
// CHECK-NEXT: [[PIPE_CONSUMER:%.*]] = xemachine.add [[PIPE_PRODUCER]], [[PIPE_PRODUCER]]

// CHECK-LABEL: func.func @pressure_guard
// CHECK: [[PRESSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[PRESSURE_CONSUMER:%.*]] = xemachine.add [[PRESSURE_PRODUCER]], [[PRESSURE_PRODUCER]]
// CHECK-NEXT: [[PRESSURE_WIDE:%.*]] = xemachine.mov
// CHECK-NEXT: xemachine.send {{.*}}[[PRESSURE_WIDE]]

// CHECK-LABEL: func.func @alias_aware_pressure_guard
// CHECK: [[ALIAS_PRESSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALIAS_PRESSURE_CONSUMER:%.*]] = xemachine.add [[ALIAS_PRESSURE_PRODUCER]],
// CHECK-NEXT: [[ALIAS_PRESSURE_CANDIDATE:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @component_pressure_guard
// CHECK: [[COMPONENT_PRESSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[COMPONENT_PRESSURE_HIGH:%.*]] = xemachine.mov [[COMPONENT_PRESSURE_PRODUCER]]
// CHECK-NEXT: [[COMPONENT_PRESSURE_TUPLE:%.*]] = xemachine.tuple_from_elements
// CHECK-NEXT: xemachine.send {{.*}}[[COMPONENT_PRESSURE_TUPLE]]
// CHECK-NEXT: [[COMPONENT_PRESSURE_CANDIDATE:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @fixed_component_pressure_guard
// CHECK: [[FIXED_PRESSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[FIXED_PRESSURE_CONSUMER:%.*]] = xemachine.add [[FIXED_PRESSURE_PRODUCER]],
// CHECK-NEXT: [[FIXED_PRESSURE_CANDIDATE:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @cross_region_pressure_guard
// CHECK: xemachine.uniform_loop
// CHECK: [[CROSS_PRESSURE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[CROSS_PRESSURE_CONSUMER:%.*]] = xemachine.add [[CROSS_PRESSURE_PRODUCER]],
// CHECK-NEXT: [[CROSS_PRESSURE_CANDIDATE:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @send_order_gap
// CHECK: [[GAP_FIRST_DST:%.*]], [[GAP_FIRST_TOKEN:%.*]] = xemachine.send
// CHECK-COUNT-8: xemachine.mov
// CHECK: [[GAP_SECOND_DST:%.*]], [[GAP_SECOND_TOKEN:%.*]] = xemachine.send {{.*}} dep [[GAP_FIRST_TOKEN]]
// CHECK-NEXT: xemachine.mov

// CHECK-LABEL: func.func @send_payload_pressure_guard
// CHECK: [[PAYLOAD:%.*]] = xemachine.mov
// CHECK-NEXT: [[PAYLOAD_FIRST_DST:%.*]], [[PAYLOAD_FIRST_TOKEN:%.*]] = xemachine.send {{.*}}[[PAYLOAD]]
// CHECK-NEXT: [[PAYLOAD_SECOND_DST:%.*]], [[PAYLOAD_SECOND_TOKEN:%.*]] = xemachine.send {{.*}} dep [[PAYLOAD_FIRST_TOKEN]]
// CHECK-NEXT: [[PAYLOAD_CANDIDATE:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @unordered_memory
// CHECK: [[UNORDERED_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[INDEPENDENT_DST:%.*]], [[INDEPENDENT_TOKEN:%.*]] = xemachine.send ugm {{.*}}desc = 33554434
// CHECK-NEXT: [[DEPENDENT_ADDRESS:%.*]] = xemachine.add [[UNORDERED_PRODUCER]], [[UNORDERED_PRODUCER]]
// CHECK-NEXT: [[BLOCKED_DST:%.*]], [[BLOCKED_TOKEN:%.*]] = xemachine.send ugm [[DEPENDENT_ADDRESS]] {{.*}}desc = 33554432

// CHECK-LABEL: func.func @token_ordered_memory
// CHECK: [[ORDERED_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[ORDERED_ADDRESS:%.*]] = xemachine.add [[ORDERED_PRODUCER]], [[ORDERED_PRODUCER]]
// CHECK-NEXT: [[FIRST_DST:%.*]], [[FIRST_TOKEN:%.*]] = xemachine.send ugm [[ORDERED_ADDRESS]] {{.*}}desc = 33554432
// CHECK-NEXT: [[SECOND_DST:%.*]], [[SECOND_TOKEN:%.*]] = xemachine.send ugm {{.*}} dep [[FIRST_TOKEN]] {{.*}}desc = 33554434

// CHECK-LABEL: func.func @arf_war
// CHECK: [[WAR_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[WAR_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[WAR_WRITER:%.*]] = xemachine.and

// CHECK-LABEL: func.func @arf_waw
// CHECK: [[WAW_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[WAW_WRITER0:%.*]] = xemachine.mov
// CHECK-NEXT: [[WAW_WRITER1:%.*]] = xemachine.and

// CHECK-LABEL: func.func @arf_raw
// CHECK: [[RAW_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[RAW_WRITER:%.*]] = xemachine.mov
// CHECK-NEXT: [[RAW_READER:%.*]] = xemachine.and

// CHECK-LABEL: func.func @virtual_arf_files
// CHECK: [[VIRTUAL_ARF_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[VIRTUAL_ARF_WRITER:%.*]] = xemachine.and
// CHECK-NEXT: [[VIRTUAL_ARF_READER:%.*]] = xemachine.add

// CHECK-LABEL: func.func @fixed_grf_war
// CHECK: [[GRF_WAR_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[GRF_WAR_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[GRF_WAR_WRITER:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @fixed_grf_waw
// CHECK: [[GRF_WAW_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[GRF_WAW_WRITER0:%.*]] = xemachine.mov
// CHECK-NEXT: [[GRF_WAW_WRITER1:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @fixed_grf_raw
// CHECK: [[GRF_RAW_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[GRF_RAW_WRITER:%.*]] = xemachine.mov
// CHECK-NEXT: [[GRF_RAW_READER:%.*]] = xemachine.and

// CHECK-LABEL: func.func @fixed_grf_tuple_alias_war
// CHECK: [[ALIAS_WAR_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALIAS_WAR_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALIAS_WAR_WRITER:%.*]] = xemachine.mov
// CHECK-NEXT: xemachine.tuple_from_elements [[ALIAS_WAR_WRITER]]

// CHECK-LABEL: func.func @destructive_tuple
// CHECK: [[TUPLE_BASE:%.*]] = xemachine.mov
// CHECK-NEXT: [[TUPLE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[TUPLE_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[TUPLE_REPLACEMENT:%.*]] = xemachine.mov
// CHECK-NEXT: [[TUPLE_UPDATED:%.*]] = xemachine.update_tuple

// CHECK-LABEL: func.func @destructive_tuple_alias_chain
// CHECK: [[CHAIN_BASE:%.*]] = xemachine.mov
// CHECK-NEXT: [[CHAIN_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[CHAIN_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[CHAIN_LOW:%.*]] = xemachine.mov
// CHECK-NEXT: [[CHAIN_HIGH:%.*]] = xemachine.mov
// CHECK-NEXT: [[CHAIN_REPLACEMENT:%.*]] = xemachine.tuple_from_elements

// CHECK-LABEL: func.func @destructive_tuple_across_region
// CHECK: [[CROSS_BASE:%.*]] = xemachine.mov
// CHECK-NEXT: [[CROSS_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[CROSS_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[CROSS_LOW:%.*]] = xemachine.mov
// CHECK-NEXT: [[CROSS_HIGH:%.*]] = xemachine.mov
// CHECK-NEXT: [[CROSS_REPLACEMENT:%.*]] = xemachine.tuple_from_elements
// CHECK-NEXT: xemachine.uniform_loop

// CHECK-LABEL: func.func @interleaved_loop_dpas_chains
// CHECK: xemachine.uniform_loop
// CHECK: ^bb0([[LOOP_DPAS_ACC0:%.*]]: {{.*}}, [[LOOP_DPAS_ACC1:%.*]]: {{.*}}):
// CHECK-NEXT: [[LOOP_DPAS_CHAIN0_FIRST:%.*]] = xemachine.dpas {{.*}}, [[LOOP_DPAS_ACC0]]
// CHECK-NEXT: [[LOOP_DPAS_CHAIN1_FIRST:%.*]] = xemachine.dpas {{.*}}, [[LOOP_DPAS_ACC1]]
// CHECK-NEXT: [[LOOP_DPAS_CHAIN0_SECOND:%.*]] = xemachine.dpas {{.*}}, [[LOOP_DPAS_CHAIN0_FIRST]]
// CHECK-NEXT: [[LOOP_DPAS_CHAIN1_SECOND:%.*]] = xemachine.dpas {{.*}}, [[LOOP_DPAS_CHAIN1_FIRST]]

// CHECK-LABEL: func.func @loop_carry
// CHECK: xemachine.uniform_loop
// CHECK: [[LOOP_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[LOOP_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[LOOP_NEXT:%.*]] = xemachine.mov

// CHECK-LABEL: func.func @loop_alias_carry
// CHECK: xemachine.uniform_loop
// CHECK: [[ALIAS_LOOP_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALIAS_LOOP_READER:%.*]] = xemachine.add
// CHECK-NEXT: [[ALIAS_LOOP_NEXT0:%.*]] = xemachine.mov
// CHECK-NEXT: [[ALIAS_LOOP_NEXT1:%.*]] = xemachine.mov
// CHECK-NEXT: [[ALIAS_LOOP_NEXT:%.*]] = xemachine.tuple_from_elements

// CHECK-LABEL: func.func @eot_boundary
// CHECK: [[EOT_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[EOT_CONSUMER:%.*]] = xemachine.add
// CHECK-NEXT: xemachine.eot

// CHECK-LABEL: func.func @raw_eot_boundary
// CHECK: [[RAW_EOT_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[RAW_EOT_CONSUMER:%.*]] = xemachine.add
// CHECK-NEXT: xemachine.send {{.*}}eot

// CHECK-LABEL: func.func @nested_region
// CHECK: [[PRE_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[PRE_FILLER:%.*]] = xemachine.mov
// CHECK-NEXT: [[PRE_CONSUMER:%.*]] = xemachine.add [[PRE_PRODUCER]], [[PRE_PRODUCER]]
// CHECK: xemachine.uniform_loop
// CHECK: [[NESTED_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[NESTED_FILLER:%.*]] = xemachine.mov
// CHECK-NEXT: [[NESTED_CONSUMER:%.*]] = xemachine.add [[NESTED_PRODUCER]], [[NESTED_PRODUCER]]
// CHECK: [[POST_PRODUCER:%.*]] = xemachine.add
// CHECK-NEXT: [[POST_FILLER:%.*]] = xemachine.mov
// CHECK-NEXT: [[POST_CONSUMER:%.*]] = xemachine.add [[POST_PRODUCER]], [[POST_PRODUCER]]
