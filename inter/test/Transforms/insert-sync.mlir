// RUN: inter-opt %s --inter-insert-sync | FileCheck %s
// RUN: inter-opt %s --inter-insert-sync -o %t
// RUN: inter-opt %t --inter-insert-sync | diff %t -

// A load destination must complete before an ALU consumer reads it.
// CHECK-LABEL: func.func @load_consumer
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.add {{.*}}swsbDistance = 1
func.func @load_consumer() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %zero = xemachine.imm 0 : i32
  %previous = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<32, 4>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %sum = xemachine.add %loaded, %loaded {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
      -> !xemachine.reg<32, 6>
  return
}

// A source-only store dependency requires source-read retirement before the
// dependent message issues.
// CHECK-LABEL: func.func @store_order
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: {{%.*}}, {{%.*}} = xemachine.load_a64 {{.*}} dep [[STORE]]
func.func @store_order() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %store
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Reusing a physical payload register waits for outstanding send source reads.
// CHECK-LABEL: func.func @payload_reuse
// CHECK: xemachine.store_a64
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.mov
func.func @payload_reuse() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 4>)
      -> !xemachine.mem.token
  %zero = xemachine.imm 0 : i32
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}

// A new physical definition cannot clobber an outstanding destination.
// CHECK-LABEL: func.func @destination_reuse
// CHECK: xemachine.load_a64
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.mov
func.func @destination_reuse() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %zero = xemachine.imm 0 : i32
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}

// Multiple destination completions share one selective allwr instruction.
// CHECK-LABEL: func.func @multiple_destination_waits
// CHECK: xemachine.load_a64
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.sync allwr {{.*}}sbidMask = 3
// CHECK-NEXT: xemachine.add
func.func @multiple_destination_waits() {
  %root = xemachine.token
  %address0 = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %address1 = xemachine.archreg 22 : !xemachine.reg<64, 22>
  %loaded0, %token0 = xemachine.load_a64 %address0 dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<16, 4>, !xemachine.mem.token)
  %loaded1, %token1 = xemachine.load_a64 %address1 dep %root
      : !xemachine.reg<64, 22> -> (!xemachine.reg<16, 6>, !xemachine.mem.token)
  %sum = xemachine.add %loaded0, %loaded1
      : (!xemachine.reg<16, 4>, !xemachine.reg<16, 6>, i32)
      -> !xemachine.reg<16, 8>
  return
}

// Distance assignment follows every operand of zero-cost tuple views.
// CHECK-LABEL: func.func @tuple_distance
// CHECK: [[LOW:%.*]] = xemachine.mov
// CHECK-NEXT: [[HIGH:%.*]] = xemachine.mov
// CHECK-NEXT: [[TUPLE:%.*]] = xemachine.tuple_from_elements [[LOW]], [[HIGH]]
// CHECK-NEXT: xemachine.mov [[TUPLE]] {{.*}}swsbDistance = 1
func.func @tuple_distance() {
  %zero = xemachine.imm 0 : i32
  %low = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  %high = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 5>
  %tuple = xemachine.tuple_from_elements %low, %high
      : (!xemachine.reg<16, 4>, !xemachine.reg<16, 5>)
      -> !xemachine.reg<32, 4>
  %copy = xemachine.mov %tuple {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, i32) -> !xemachine.reg<32, 6>
  return
}

// DPAS uses its asynchronous token and materializes an ALU input distance on a
// preceding sync because DPAS cannot encode both.
// CHECK-LABEL: func.func @dpas_token_only
// CHECK: xemachine.mov
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbDistance = 1
// CHECK: xemachine.dpas
// CHECK-SAME: swsbTokenMode = 1
// CHECK-NOT: swsbDistance
// CHECK: return
func.func @dpas_token_only() {
  %a = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %b = xemachine.archreg 24 : !xemachine.reg<128, 24>
  %zero = xemachine.imm 0 : f32
  %acc = xemachine.mov %zero {execSize = 16 : i32}
      : (!xemachine.imm, f32) -> !xemachine.reg<128, 32>
  %result = xemachine.dpas %a, %b, %acc {
      aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
      : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
         !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  return
}

// A DPAS may need both an ALU input distance and a send-result token wait. The
// two preceding syncs remain stable when the pass is repeated.
// CHECK-LABEL: func.func @dpas_distance_and_token_wait
// CHECK: xemachine.load_a64
// CHECK: xemachine.mov
// CHECK: xemachine.sync nop {{.*}}swsbDistance = 1
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.dpas
func.func @dpas_distance_and_token_wait() {
  %root = xemachine.token
  %address = xemachine.archreg 40 : !xemachine.reg<64, 40>
  %b = xemachine.archreg 24 : !xemachine.reg<128, 24>
  %loaded, %token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 40>
      -> (!xemachine.reg<128, 32>, !xemachine.mem.token)
  %zero = xemachine.imm 0 : i32
  %a = xemachine.mov %zero {execSize = 16 : i32}
      : (!xemachine.imm, i32) -> !xemachine.reg<64, 20>
  %result = xemachine.dpas %a, %b, %loaded {
      aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
      : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
         !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  return
}

// Exact SSA consumers still classify virtual message destinations as writes.
// CHECK-LABEL: func.func @virtual_destination
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.add
func.func @virtual_destination() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
  %sum = xemachine.add %loaded, %loaded {execSize = 32 : i32}
      : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>, i32)
      -> !xemachine.reg<32, -1>
  return
}

// An issue-only token intentionally drops completion state.
// CHECK-LABEL: func.func @issue_only
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: [[AFTER:%.*]] = xemachine.after [[STORE]]
// CHECK-NOT: xemachine.sync
// CHECK: xemachine.load_a64 {{.*}} dep [[AFTER]]
func.func @issue_only() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %after = xemachine.after %store : !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %after
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// sync.bar synchronizes threads; it does not retire outstanding writes.
// CHECK-LABEL: func.func @bar_preserves_scoreboard
// CHECK: {{%.*}}, [[LOAD:%.*]] = xemachine.load_a64
// CHECK-NEXT: [[BAR:%.*]] = xemachine.sync bar dep [[LOAD]]
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.add
func.func @bar_preserves_scoreboard() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %bar = xemachine.sync bar dep %load_token : !xemachine.mem.token
  %sum = xemachine.add %loaded, %loaded {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
      -> !xemachine.reg<32, 6>
  return
}

// sync.bar retires source reads from sends issued before the barrier.
// CHECK-LABEL: func.func @bar_retires_reads
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: [[BAR:%.*]] = xemachine.sync bar dep [[STORE]]
// CHECK-NEXT: {{%.*}}, {{%.*}} = xemachine.load_a64 {{.*}} dep [[BAR]]
func.func @bar_retires_reads() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %bar = xemachine.sync bar dep %store : !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %bar
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// CHECK-LABEL: func.func @joined_eot
// CHECK: {{%.*}}, [[LOAD:%.*]] = xemachine.load_a64
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK: [[JOIN:%.*]] = xemachine.token_join [[LOAD]], [[STORE]]
// CHECK-NEXT: {{%.*}} = xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: {{%.*}} = xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.eot {{.*}} dep [[JOIN]]
// CHECK-NEXT: return
func.func @joined_eot() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loaded, %load = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %join = xemachine.token_join %load, %store
      : !xemachine.mem.token, !xemachine.mem.token
  xemachine.eot %data dep %join : !xemachine.reg<16, 2>
  return
}

// Region exits preserve pending completion until the continuation consumes it.
// CHECK-LABEL: func.func @branch_join
// CHECK: [[ROOT:%.*]] = xemachine.token
// CHECK: [[IF:%.*]] = xemachine.exec_if
// CHECK: [[STORE:%.*]] = xemachine.store_a64 {{.*}} dep [[ROOT]]
// CHECK: xemachine.yield [[STORE]]
// CHECK: xemachine.yield [[ROOT]]
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[IF]]
func.func @branch_join(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %if_token = xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
    %store = xemachine.store_a64 %address data %data dep %root
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.yield %store : !xemachine.mem.token
  } otherwise {
    xemachine.yield %root : !xemachine.mem.token
  } -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %if_token
      : !xemachine.reg<64, 20> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Yield is bookkeeping, so an escaping load waits at its real continuation
// consumer rather than at the region terminator.
// CHECK-LABEL: func.func @branch_data_result
// CHECK: [[RESULTS:%.*]]:2 = xemachine.uniform_if
// CHECK: {{%.*}}, {{%.*}} = xemachine.load_a64
// CHECK-NEXT: xemachine.yield
// CHECK: } -> !xemachine.reg<32, 4>, !xemachine.mem.token
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.add [[RESULTS]]#0, [[RESULTS]]#0
func.func @branch_data_result(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data_result, %token_result =
      xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<64, 20>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    xemachine.yield %loaded, %load
        : !xemachine.reg<32, 4>, !xemachine.mem.token
  } otherwise {
    %zero = xemachine.imm 0 : i32
    %fallback = xemachine.mov %zero {execSize = 32 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, 4>
    xemachine.yield %fallback, %root
        : !xemachine.reg<32, 4>, !xemachine.mem.token
  } -> !xemachine.reg<32, 4>, !xemachine.mem.token
  %sum = xemachine.add %data_result, %data_result {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
      -> !xemachine.reg<32, 6>
  return
}

// The absent else edge preserves the incoming scoreboard state.
// CHECK-LABEL: func.func @branch_without_else
// CHECK: xemachine.store_a64
// CHECK: xemachine.exec_if
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK: xemachine.yield
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
func.func @branch_without_else(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
    %zero = xemachine.imm 0 : i32
    %then_reuse = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 2>
    xemachine.yield
  }
  %zero = xemachine.imm 0 : i32
  %continuation_reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 2>
  return
}

// The exec_if region graph carries then-arm state into the sequential else arm.
// CHECK-LABEL: func.func @cross_arm_hazards
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.store_a64
// CHECK-NEXT: xemachine.yield
// CHECK-NEXT: } otherwise {
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.mov {{.*}}noMask{{.*}} -> !xemachine.reg<16, 4>
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.mov {{.*}}noMask{{.*}} -> !xemachine.reg<16, 6>
func.func @cross_arm_hazards(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %payload = xemachine.archreg 6 : !xemachine.reg<16, 6>
  %zero = xemachine.imm 0 : i32
  xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<64, 20>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    %store = xemachine.store_a64 %address data %payload dep %root
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 6>)
        -> !xemachine.mem.token
    xemachine.yield
  } otherwise {
    %reuse_destination = xemachine.mov %zero {noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 4>
    %reuse_source = xemachine.mov %zero {noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 6>
    xemachine.yield
  }
  return
}

// Multiple result tokens converge across repeated analysis and rewrite runs.
// A mixed load/store source remains pending after allwr and still needs allrd.
// CHECK-LABEL: func.func @multiple_branch_tokens
// CHECK: [[ZERO:%.*]] = xemachine.imm 0
// CHECK: {{%.*}}:2 = xemachine.uniform_if
// CHECK: } -> !xemachine.mem.token, !xemachine.mem.token
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.mov [[ZERO]] {{.*}} -> !xemachine.reg<16, 4>
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.mov [[ZERO]] {{.*}} -> !xemachine.reg<64, 20>
func.func @multiple_branch_tokens(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %zero = xemachine.imm 0 : i32
  %load_result, %store_result =
      xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<64, 20>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    %store = xemachine.store_a64 %address data %data dep %root
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.yield %load, %store
        : !xemachine.mem.token, !xemachine.mem.token
  } otherwise {
    xemachine.yield %root, %root
        : !xemachine.mem.token, !xemachine.mem.token
  } -> !xemachine.mem.token, !xemachine.mem.token
  %reuse_destination = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  %reuse_address = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<64, 20>
  return
}

// Loop-carried token completion reaches both the backedge and loop result.
// CHECK-LABEL: func.func @loop_carried_token
// CHECK: [[LOOP:%.*]] = xemachine.uniform_loop
// CHECK: ^bb0([[ITER:%.*]]: !xemachine.mem.token):
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: [[STORE:%.*]] = xemachine.store_a64 {{.*}} dep [[ITER]]
// CHECK-NEXT: xemachine.continue_if {{.*}}([[STORE]] : !xemachine.mem.token)
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[LOOP]]
func.func @loop_carried_token(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loop_token = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    %store = xemachine.store_a64 %address data %data dep %iter
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%store : !xemachine.mem.token)
  } : (!xemachine.mem.token) -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %loop_token
      : !xemachine.reg<64, 20>
      -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Store-only loop backedges retire the source phase at the next issue.
// CHECK-LABEL: func.func @loop_store_only
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.store_a64
// CHECK-NEXT: xemachine.continue_if
func.func @loop_store_only(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  xemachine.uniform_loop () {
  ^bb0:
    %store = xemachine.store_a64 %address data %data dep %root
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
  } : () -> ()
  return
}

// Nested region interfaces participate in the enclosing loop fixed point.
// CHECK-LABEL: func.func @nested_loop_branch
// CHECK: [[LOOP:%.*]] = xemachine.uniform_loop
// CHECK: [[BRANCH:%.*]] = xemachine.uniform_if
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK: xemachine.continue_if
// CHECK: xemachine.eot {{.*}} dep [[LOOP]]
func.func @nested_loop_branch(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loop_token = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    %branch_token =
        xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      %store = xemachine.store_a64 %address data %data dep %iter
          : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
          -> !xemachine.mem.token
      xemachine.yield %store : !xemachine.mem.token
    } otherwise {
      xemachine.yield %iter : !xemachine.mem.token
    } -> !xemachine.mem.token
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%branch_token : !xemachine.mem.token)
  } : (!xemachine.mem.token) -> !xemachine.mem.token
  xemachine.eot %data dep %loop_token : !xemachine.reg<16, 2>
  return
}

// Region entry operands are forwarded, not consumed by the branch operation.
// The wait belongs at the first real use of the corresponding block argument.
// CHECK-LABEL: func.func @loop_forwarded_init
// CHECK: {{%.*}}, {{%.*}} = xemachine.load_a64
// CHECK-NEXT: [[LOOP:%.*]] = xemachine.uniform_loop
// CHECK: ^bb0([[ITER:%.*]]: !xemachine.reg<32, 4>):
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.add [[ITER]], [[ITER]]
func.func @loop_forwarded_init(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<64, 20>
      -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %loop_result = xemachine.uniform_loop (%loaded) {
  ^bb0(%iter: !xemachine.reg<32, 4>):
    %sum = xemachine.add %iter, %iter {execSize = 32 : i32}
        : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
        -> !xemachine.reg<32, 6>
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%iter : !xemachine.reg<32, 4>)
  } : (!xemachine.reg<32, 4>) -> !xemachine.reg<32, 4>
  return
}

// The analysis is driven by RegionBranchOpInterface, not XeMachine op names.
// CHECK-LABEL: func.func @generic_region_branch_loop
// CHECK: [[LOOP:%.*]] = scf.for
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: scf.yield [[STORE]]
// CHECK: xemachine.sync nop {{.*}}swsbTokenMode = 2
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[LOOP]]
func.func @generic_region_branch_loop() {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %loop_token = scf.for %iv = %lb to %ub step %step
      iter_args(%iter = %root) -> !xemachine.mem.token {
    %store = xemachine.store_a64 %address data %data dep %iter
        : (!xemachine.reg<64, 20>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    scf.yield %store : !xemachine.mem.token
  }
  %loaded, %load_token = xemachine.load_a64 %address dep %loop_token
      : !xemachine.reg<64, 20>
      -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Solver states conservatively retain inferred drains. Local rewrite replay
// must still preserve a nested-region issue discovered after loop convergence.
// CHECK-LABEL: func.func @nested_replay_after_fixpoint
// CHECK: xemachine.uniform_if
// CHECK: {{%.*}}, {{%.*}} = xemachine.load_a64 {{.*}} -> (!xemachine.reg<32, 8>, !xemachine.mem.token)
// CHECK: xemachine.yield
// CHECK-NEXT: }
// CHECK: xemachine.mov {{.*}} -> !xemachine.reg<16, 8>
func.func @nested_replay_after_fixpoint(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %zero = xemachine.imm 0 : i32
  %loop = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      %x, %x_token = xemachine.load_a64 %address dep %root
          : !xemachine.reg<64, 20>
          -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
      %a, %a_token = xemachine.load_a64 %address dep %iter
          : !xemachine.reg<64, 20>
          -> (!xemachine.reg<32, 12>, !xemachine.mem.token)
      %w, %w_token = xemachine.load_a64 %address dep %root
          : !xemachine.reg<64, 20>
          -> (!xemachine.reg<32, 8>, !xemachine.mem.token)
      %sum = xemachine.add %x, %x {execSize = 32 : i32}
          : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
          -> !xemachine.reg<32, 20>
      xemachine.yield
    }
    %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 8>
    %z, %z_token = xemachine.load_a64 %address dep %root
        : !xemachine.reg<64, 20>
        -> (!xemachine.reg<32, 16>, !xemachine.mem.token)
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%z_token : !xemachine.mem.token)
  } : (!xemachine.mem.token) -> !xemachine.mem.token
  return
}

// A physical destination from the previous iteration reaches the backedge
// even without an SSA token carry.
// CHECK-LABEL: func.func @loop_physical_waw
// CHECK: xemachine.uniform_loop
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.load_a64
// CHECK-NEXT: xemachine.continue_if
// CHECK: }
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: xemachine.mov
func.func @loop_physical_waw(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %zero = xemachine.imm 0 : i32
  xemachine.uniform_loop () {
  ^bb0:
    %loaded, %load_token = xemachine.load_a64 %address dep %root
        : !xemachine.reg<64, 20>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
  } : () -> ()
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}

// A loop-carried DPAS remains pending across the backedge. Its wait belongs at
// the next accumulator use, leaving independent tail work exposed.
// CHECK-LABEL: func.func @loop_carried_dpas
// CHECK: ^bb0([[ACC:%.*]]: !xemachine.reg<128, 32>):
// CHECK-NEXT: xemachine.sync nop {{.*}}swsbTokenMode = 3
// CHECK-NEXT: [[DPAS:%.*]] = xemachine.dpas {{.*}}, [[ACC]]
// CHECK-NEXT: xemachine.mov
// CHECK-NEXT: xemachine.continue_if {{.*}}([[DPAS]] : !xemachine.reg<128, 32>)
func.func @loop_carried_dpas(%flag: !xemachine.arf<f, 2, 0>) {
  %a = xemachine.archreg 20 : !xemachine.reg<64, 20>
  %b = xemachine.archreg 24 : !xemachine.reg<128, 24>
  %acc = xemachine.archreg 32 : !xemachine.reg<128, 32>
  %zero = xemachine.imm 0 : i32
  %result = xemachine.uniform_loop (%acc) {
  ^bb0(%iter: !xemachine.reg<128, 32>):
    %next = xemachine.dpas %a, %b, %iter {
        aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
        : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>,
           !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
    %tail = xemachine.mov %zero
        : (!xemachine.imm, i32) -> !xemachine.reg<16, 4>
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%next : !xemachine.reg<128, 32>)
  } : (!xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  return
}

// Mutually exclusive uniform arms inherit the same parent state, not each
// other's lexical instruction history.
// CHECK-LABEL: func.func @distance_uniform_arms
// CHECK: xemachine.uniform_if
// CHECK: xemachine.mov {{.*}} -> !xemachine.reg<16, 6>
// CHECK: } otherwise {
// CHECK: xemachine.mov {{.*}} -> !xemachine.reg<16, 8>
// CHECK-NOT: swsbDistance
// CHECK: }
func.func @distance_uniform_arms(%flag: !xemachine.arf<f, 2, 0>) {
  %zero = xemachine.imm 0 : i32
  xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
    %then = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 6>
    xemachine.yield
  } otherwise {
    %else = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 8>
    xemachine.yield
  }
  return
}

// A loop backedge carries the previous iteration's physical ALU write.
// CHECK-LABEL: func.func @distance_loop_backedge
// CHECK: xemachine.uniform_loop
// CHECK: xemachine.mov {{.*}}swsbDistance = 1
func.func @distance_loop_backedge(%flag: !xemachine.arf<f, 2, 0>) {
  %storage = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %zero = xemachine.imm 0 : i32
  xemachine.uniform_loop () {
  ^bb0:
    %read = xemachine.mov %storage : (!xemachine.reg<16, 4>, i32)
        -> !xemachine.reg<16, 6>
    %write = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 4>
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
  } : () -> ()
  return
}

// Region footprints distinguish GRFs within one multi-GRF allocation.
// CHECK-LABEL: func.func @distance_disjoint_grfs
// CHECK: xemachine.mov
// CHECK-NEXT: xemachine.mov
// CHECK-NOT: swsbDistance
// CHECK: return
func.func @distance_disjoint_grfs() {
  %zero = xemachine.imm 0 : i32
  %low = xemachine.mov %zero {execSize = 1 : i32}
      : (!xemachine.imm, i32) -> !xemachine.reg<32, 4>
  %high = xemachine.mov %low {execSize = 1 : i32, src0Sub = 16 : i32}
      : (!xemachine.reg<32, 4>, i32) -> !xemachine.reg<1, 8>
  return
}

// Subregisters in one GRF share its hardware dependency bucket.
// CHECK-LABEL: func.func @distance_same_grf
// CHECK: xemachine.mov
// CHECK-NEXT: xemachine.mov {{.*}}swsbDistance = 1
func.func @distance_same_grf() {
  %zero = xemachine.imm 0 : i32
  %low = xemachine.mov %zero {execSize = 1 : i32}
      : (!xemachine.imm, i32) -> !xemachine.reg<16, 4>
  %high = xemachine.mov %low {execSize = 1 : i32, src0Sub = 1 : i32}
      : (!xemachine.reg<16, 4>, i32) -> !xemachine.reg<1, 8>
  return
}

// A type conversion crossing ALU pipes requires an all-pipe distance.
// CHECK-LABEL: func.func @distance_cross_pipe
// CHECK: xemachine.mov
// CHECK-NEXT: xemachine.mov {{.*}}swsbDistance = 1{{.*}}swsbPipe = 1
func.func @distance_cross_pipe() {
  %zero = xemachine.imm 0 : f32
  %float = xemachine.mov %zero {execSize = 16 : i32}
      : (!xemachine.imm, f32) -> !xemachine.reg<16, 4>
  %integer = xemachine.mov %float {execSize = 16 : i32, src0Type = f32}
      : (!xemachine.reg<16, 4>, i32) -> !xemachine.reg<16, 8>
  return
}

// The payload setup may execute or be bypassed, so continuation state merges
// both paths instead of assuming the nested body always ran.
// CHECK-LABEL: func.func @distance_payload_bypass
// CHECK: xemachine.payload_prologue
// CHECK: xemachine.mov
// CHECK: }
// CHECK-NEXT: xemachine.mov {{.*}}swsbDistance = 1
// CHECK: return
func.func @distance_payload_bypass() {
  %zero = xemachine.imm 0 : i32
  %storage = xemachine.archreg 4 : !xemachine.reg<16, 4>
  xemachine.payload_prologue {
    %setup = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 4>
    xemachine.payload_prologue_end
  }
  %continuation = xemachine.mov %storage : (!xemachine.reg<16, 4>, i32)
      -> !xemachine.reg<16, 8>
  return
}
