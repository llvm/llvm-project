// RUN: inter-opt %s --inter-insert-sync | FileCheck %s
// RUN: inter-opt %s --inter-insert-sync -o %t
// RUN: inter-opt %t --inter-insert-sync | diff %t -

// A load destination must complete before an ALU consumer reads it.
// CHECK-LABEL: func.func @load_consumer
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.add
func.func @load_consumer() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %sum = xemachine.add %loaded, %loaded {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
      -> !xemachine.reg<32, 6>
  return
}

// A source-only store dependency requires source-read retirement before the
// dependent message issues.
// CHECK-LABEL: func.func @store_order
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: xemachine.sync allrd
// CHECK-NEXT: {{%.*}}, {{%.*}} = xemachine.load_a64 {{.*}} dep [[STORE]]
func.func @store_order() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %store
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Reusing a physical payload register waits for outstanding send source reads.
// CHECK-LABEL: func.func @payload_reuse
// CHECK: xemachine.store_a64
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.mov
func.func @payload_reuse() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<16, 1>, !xemachine.reg<16, 4>)
      -> !xemachine.mem.token
  %zero = xemachine.imm 0 : i32
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}

// A new physical definition cannot clobber an outstanding destination.
// CHECK-LABEL: func.func @destination_reuse
// CHECK: xemachine.load_a64
// CHECK: xemachine.sync allwr
// CHECK-NEXT: xemachine.mov
func.func @destination_reuse() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %zero = xemachine.imm 0 : i32
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}

// Exact SSA consumers still classify virtual message destinations as writes.
// CHECK-LABEL: func.func @virtual_destination
// CHECK: xemachine.load_a64
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.add
func.func @virtual_destination() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
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
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %after = xemachine.after %store : !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %after
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// sync.bar synchronizes threads; it does not retire outstanding writes.
// CHECK-LABEL: func.func @bar_preserves_scoreboard
// CHECK: {{%.*}}, [[LOAD:%.*]] = xemachine.load_a64
// CHECK-NEXT: [[BAR:%.*]] = xemachine.sync bar dep [[LOAD]]
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.add
func.func @bar_preserves_scoreboard() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %bar = xemachine.sync bar dep %load_token : !xemachine.mem.token
  %sum = xemachine.add %loaded, %loaded {execSize = 32 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
      -> !xemachine.reg<32, 6>
  return
}

// CHECK-LABEL: func.func @joined_eot
// CHECK: {{%.*}}, [[LOAD:%.*]] = xemachine.load_a64
// CHECK: [[STORE:%.*]] = xemachine.store_a64
// CHECK: [[JOIN:%.*]] = xemachine.token_join [[LOAD]], [[STORE]]
// CHECK-NEXT: [[READ:%.*]] = xemachine.sync allrd
// CHECK-NEXT: [[WRITE:%.*]] = xemachine.sync allwr dep [[READ]]
// CHECK-NEXT: xemachine.eot {{.*}} dep [[JOIN]]
func.func @joined_eot() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loaded, %load = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
      -> !xemachine.mem.token
  %join = xemachine.token_join %load, %store
      : !xemachine.mem.token, !xemachine.mem.token
  xemachine.eot %data dep %join : !xemachine.reg<16, 2>
  return
}

// Region exits are joined before synchronization at the continuation.
// CHECK-LABEL: func.func @branch_join
// CHECK: [[ROOT:%.*]] = xemachine.token
// CHECK: [[IF:%.*]] = xemachine.exec_if
// CHECK: [[STORE:%.*]] = xemachine.store_a64 {{.*}} dep [[ROOT]]
// CHECK: xemachine.yield [[STORE]]
// CHECK: xemachine.yield [[ROOT]]
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[IF]]
func.func @branch_join(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %if_token = xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
    %store = xemachine.store_a64 %address data %data dep %root
        : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.yield %store : !xemachine.mem.token
  } otherwise {
    xemachine.yield %root : !xemachine.mem.token
  } -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %if_token
      : !xemachine.reg<16, 1> -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Yield is bookkeeping, so an escaping load waits at its real continuation
// consumer rather than at the region terminator.
// CHECK-LABEL: func.func @branch_data_result
// CHECK: [[RESULTS:%.*]]:2 = xemachine.uniform_if
// CHECK: {{%.*}}, {{%.*}} = xemachine.load_a64
// CHECK-NEXT: xemachine.yield
// CHECK: } -> !xemachine.reg<32, 4>, !xemachine.mem.token
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.add [[RESULTS]]#0, [[RESULTS]]#0
func.func @branch_data_result(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data_result, %token_result =
      xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<16, 1>
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
// CHECK: xemachine.sync allrd
// CHECK: xemachine.yield
// CHECK: xemachine.sync allrd
func.func @branch_without_else(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %store = xemachine.store_a64 %address data %data dep %root
      : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
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
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.mov {{.*}}noMask{{.*}} -> !xemachine.reg<16, 4>
// CHECK-NEXT: xemachine.sync allrd
// CHECK-NEXT: xemachine.mov {{.*}}noMask{{.*}} -> !xemachine.reg<16, 6>
func.func @cross_arm_hazards(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %payload = xemachine.archreg 6 : !xemachine.reg<16, 6>
  %zero = xemachine.imm 0 : i32
  xemachine.exec_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<16, 1>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    %store = xemachine.store_a64 %address data %payload dep %root
        : (!xemachine.reg<16, 1>, !xemachine.reg<16, 6>)
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
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.mov [[ZERO]] {{.*}} -> !xemachine.reg<16, 4>
// CHECK-NEXT: xemachine.sync allrd
// CHECK-NEXT: xemachine.mov [[ZERO]] {{.*}} -> !xemachine.reg<16, 1>
func.func @multiple_branch_tokens(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %zero = xemachine.imm 0 : i32
  %load_result, %store_result =
      xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
    %loaded, %load = xemachine.load_a64 %address dep %root
        : !xemachine.reg<16, 1>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    %store = xemachine.store_a64 %address data %data dep %root
        : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
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
      -> !xemachine.reg<16, 1>
  return
}

// Loop-carried token completion reaches both the backedge and loop result.
// CHECK-LABEL: func.func @loop_carried_token
// CHECK: [[LOOP:%.*]] = xemachine.uniform_loop
// CHECK: ^bb0([[ITER:%.*]]: !xemachine.mem.token):
// CHECK-NEXT: xemachine.sync allrd
// CHECK-NEXT: [[STORE:%.*]] = xemachine.store_a64 {{.*}} dep [[ITER]]
// CHECK-NEXT: xemachine.continue_if {{.*}}([[STORE]] : !xemachine.mem.token)
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[LOOP]]
func.func @loop_carried_token(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loop_token = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    %store = xemachine.store_a64 %address data %data dep %iter
        : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
        (%store : !xemachine.mem.token)
  } : (!xemachine.mem.token) -> !xemachine.mem.token
  %loaded, %load_token = xemachine.load_a64 %address dep %loop_token
      : !xemachine.reg<16, 1>
      -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
  return
}

// Nested region interfaces participate in the enclosing loop fixed point.
// CHECK-LABEL: func.func @nested_loop_branch
// CHECK: [[LOOP:%.*]] = xemachine.uniform_loop
// CHECK: [[BRANCH:%.*]] = xemachine.uniform_if
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.store_a64
// CHECK: xemachine.continue_if
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.eot {{.*}} dep [[LOOP]]
func.func @nested_loop_branch(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %loop_token = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    %branch_token =
        xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      %store = xemachine.store_a64 %address data %data dep %iter
          : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
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
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.add [[ITER]], [[ITER]]
func.func @loop_forwarded_init(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %loaded, %load_token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<16, 1>
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
// CHECK: xemachine.sync allrd
// CHECK-NEXT: [[STORE:%.*]] = xemachine.store_a64
// CHECK-NEXT: scf.yield [[STORE]]
// CHECK: xemachine.sync allrd
// CHECK-NEXT: xemachine.load_a64 {{.*}} dep [[LOOP]]
func.func @generic_region_branch_loop() {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %data = xemachine.archreg 2 : !xemachine.reg<16, 2>
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %loop_token = scf.for %iv = %lb to %ub step %step
      iter_args(%iter = %root) -> !xemachine.mem.token {
    %store = xemachine.store_a64 %address data %data dep %iter
        : (!xemachine.reg<16, 1>, !xemachine.reg<16, 2>)
        -> !xemachine.mem.token
    scf.yield %store : !xemachine.mem.token
  }
  %loaded, %load_token = xemachine.load_a64 %address dep %loop_token
      : !xemachine.reg<16, 1>
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
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.mov {{.*}} -> !xemachine.reg<16, 8>
func.func @nested_replay_after_fixpoint(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %zero = xemachine.imm 0 : i32
  %loop = xemachine.uniform_loop (%root) {
  ^bb0(%iter: !xemachine.mem.token):
    xemachine.uniform_if %flag : !xemachine.arf<f, 2, 0> {
      %x, %x_token = xemachine.load_a64 %address dep %root
          : !xemachine.reg<16, 1>
          -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
      %a, %a_token = xemachine.load_a64 %address dep %iter
          : !xemachine.reg<16, 1>
          -> (!xemachine.reg<32, 12>, !xemachine.mem.token)
      %w, %w_token = xemachine.load_a64 %address dep %root
          : !xemachine.reg<16, 1>
          -> (!xemachine.reg<32, 8>, !xemachine.mem.token)
      %sum = xemachine.add %x, %x {execSize = 32 : i32}
          : (!xemachine.reg<32, 4>, !xemachine.reg<32, 4>, i32)
          -> !xemachine.reg<32, 20>
      xemachine.yield
    }
    %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, 8>
    %z, %z_token = xemachine.load_a64 %address dep %root
        : !xemachine.reg<16, 1>
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
// CHECK-NEXT: xemachine.sync allwr
// CHECK-NEXT: xemachine.load_a64
// CHECK: xemachine.continue_if
// CHECK: }
// CHECK: xemachine.sync allwr
// CHECK-NEXT: xemachine.mov
func.func @loop_physical_waw(%flag: !xemachine.arf<f, 2, 0>) {
  %root = xemachine.token
  %address = xemachine.archreg 1 : !xemachine.reg<16, 1>
  %zero = xemachine.imm 0 : i32
  xemachine.uniform_loop () {
  ^bb0:
    %loaded, %load_token = xemachine.load_a64 %address dep %root
        : !xemachine.reg<16, 1>
        -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
  } : () -> ()
  %reuse = xemachine.mov %zero : (!xemachine.imm, i32)
      -> !xemachine.reg<16, 4>
  return
}
