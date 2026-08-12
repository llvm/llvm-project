// RUN: inter-opt %s --inter-infer-memory-tokens | FileCheck %s
// RUN: inter-opt %s --inter-infer-memory-tokens --inter-infer-memory-tokens --verify-each -o /dev/null

func.func @aliasing(%input: !xw.ptr<#xw.global>,
                    %output: !xw.ptr<#xw.global>) attributes {
    xw.simd_width = 16 : i32} {
  %value, %read = xw.load %input
      : (!xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
  %written = xw.store %value -> %output
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @aliasing
// CHECK: %[[VALUE:.*]], %[[READ:.*]] = xw.load
// CHECK: xw.store %[[VALUE]] -> {{.*}} after %[[READ]]
// CHECK-SAME: xw.tokens_inferred
// CHECK-NOT: llvm.
// CHECK-NOT: cf.

// -----

func.func @if_one_branch(%condition: i1, %ptr: !xw.ptr<#xw.global>,
                         %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %before = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  scf.if %condition {
    %inside = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  }
  %after = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @if_one_branch
// CHECK: %[[BEFORE:.*]] = xw.store
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (!xw.mem.token)
// CHECK: %[[INSIDE:.*]] = xw.store {{.*}} after %[[BEFORE]]
// CHECK: scf.yield %[[INSIDE]] : !xw.mem.token
// CHECK: } else {
// CHECK: scf.yield %[[BEFORE]] : !xw.mem.token
// CHECK: xw.store {{.*}} after %[[IF]]

// -----

func.func @if_both_branches(%condition: i1, %ptr: !xw.ptr<#xw.global>,
                            %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %before = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  scf.if %condition {
    %then = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  } else {
    %else = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  }
  %after = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @if_both_branches
// CHECK: %[[BEFORE:.*]] = xw.store
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (!xw.mem.token)
// CHECK: %[[THEN:.*]] = xw.store {{.*}} after %[[BEFORE]]
// CHECK: scf.yield %[[THEN]] : !xw.mem.token
// CHECK: %[[ELSE:.*]] = xw.store {{.*}} after %[[BEFORE]]
// CHECK: scf.yield %[[ELSE]] : !xw.mem.token
// CHECK: xw.store {{.*}} after %[[IF]]

// -----

func.func @where_branches(%mask: !xw.mask<16>,
                          %ptr: !xw.ptr<#xw.global>,
                          %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %before = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  xw.where %mask {
    %then = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
    xw.yield
  } otherwise {
    xw.yield
  } : !xw.mask<16>
  %after = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @where_branches
// CHECK: %[[BEFORE:.*]] = xw.store
// CHECK: %[[WHERE:.*]] = xw.where {{.*}} {
// CHECK: %[[THEN:.*]] = xw.store {{.*}} after %[[BEFORE]]
// CHECK: xw.yield %[[THEN]] : !xw.mem.token
// CHECK: otherwise {
// CHECK: xw.yield %[[BEFORE]] : !xw.mem.token
// CHECK: } : !xw.mask<16> -> !xw.mem.token
// CHECK: xw.store {{.*}} after %[[WHERE]]

// -----

func.func @for_loop(%ptr: !xw.ptr<#xw.global>,
                    %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %before = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  scf.for %i = %c0 to %c4 step %c1 {
    %inside = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  }
  %after = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @for_loop
// CHECK: %[[BEFORE:.*]] = xw.store
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = %[[BEFORE]]) -> (!xw.mem.token)
// CHECK: %[[INSIDE:.*]] = xw.store {{.*}} after %[[ITER]]
// CHECK: scf.yield %[[INSIDE]] : !xw.mem.token
// CHECK: xw.store {{.*}} after %[[LOOP]]

// -----

func.func @while_loop(%condition: i1, %ptr: !xw.ptr<#xw.global>,
                      %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %before = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  scf.while : () -> () {
    %loaded, %test = xw.load %ptr
        : (!xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
    scf.condition(%condition)
  } do {
    %inside = xw.store %value -> %ptr
        : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
    scf.yield
  }
  %after = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @while_loop
// CHECK: %[[BEFORE:.*]] = xw.store
// CHECK: %[[LOOP:.*]] = scf.while (%[[BEFORE_ARG:.*]] = %[[BEFORE]]) : (!xw.mem.token) -> !xw.mem.token
// CHECK: %{{.*}}, %[[TEST:.*]] = xw.load {{.*}} after %[[BEFORE_ARG]]
// CHECK: scf.condition({{.*}}) %[[TEST]] : !xw.mem.token
// CHECK: ^bb0(%[[AFTER_ARG:.*]]: !xw.mem.token):
// CHECK: %[[INSIDE:.*]] = xw.store {{.*}} after %[[AFTER_ARG]]
// CHECK: scf.yield %[[INSIDE]] : !xw.mem.token
// CHECK: xw.store {{.*}} after %[[LOOP]]

// -----

func.func @nested_and_chain(%condition: i1, %ptr: !xw.ptr<#xw.global>,
                            %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %first = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  scf.if %condition {
    scf.for %i = %c0 to %c2 step %c1 {
      %nested = xw.store %value -> %ptr
          : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
    }
  }
  %second = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  %third = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @nested_and_chain
// CHECK: %[[FIRST:.*]] = xw.store
// CHECK: %[[OUTER:.*]] = scf.if
// CHECK: %[[INNER:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[FIRST]])
// CHECK: scf.yield %[[INNER]] : !xw.mem.token
// CHECK: %[[SECOND:.*]] = xw.store {{.*}} after %[[OUTER]]
// CHECK: %[[CHAIN:.*]] = xw.join %[[SECOND]], %[[OUTER]]
// CHECK: xw.store {{.*}} after %[[CHAIN]]

// -----

func.func @barrier_release_and_dedup(%ptr: !xw.ptr<#xw.global>,
                                    %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %allocation = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %stored = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  %barrier = xw.barrier %stored
      : !xw.mem.token -> !xw.mem.token
  %released = xw.alloc_release %allocation after %barrier
      : (!xw.ptr<#xw.local>, !xw.mem.token) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @barrier_release_and_dedup
// CHECK: %[[STORED:.*]] = xw.store
// CHECK-NOT: xw.join
// CHECK: %[[BARRIER:.*]] = xw.barrier %[[STORED]]
// CHECK-NOT: xw.join
// CHECK: xw.alloc_release {{.*}} after %[[BARRIER]]

// -----

func.func @atomic(%ptr: !xw.ptr<#xw.global>,
                  %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %stored = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  %old, %atomic = xw.atomic_rmw addi %value, %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>)
      -> (!xw.simd<i32, 16>, !xw.mem.token)
  return
}

// CHECK-LABEL: func.func @atomic
// CHECK: %[[STORED:.*]] = xw.store
// CHECK: xw.atomic_rmw {{.*}} after %[[STORED]]

// -----

func.func @multiple_producer_chains(%ptr: !xw.ptr<#xw.global>,
                                    %value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %external = xw.token : !xw.mem.token
  %first = xw.store %value -> %ptr
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  %second = xw.store %value -> %ptr after %external
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>, !xw.mem.token)
      -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @multiple_producer_chains
// CHECK: %[[EXTERNAL:.*]] = xw.token
// CHECK: %[[FIRST:.*]] = xw.store
// CHECK: %[[JOINED:.*]] = xw.join %[[EXTERNAL]], %[[FIRST]]
// CHECK: xw.store {{.*}} after %[[JOINED]]

// -----

func.func @independent_and_reads(%value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %a = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %b = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %store_a = xw.store %value -> %a
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.local>) -> !xw.mem.token
  %store_b = xw.store %value -> %b
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.local>) -> !xw.mem.token
  %read_a, %read_token_a = xw.load %a
      : (!xw.ptr<#xw.local>) -> (!xw.simd<i32, 16>, !xw.mem.token)
  %read_b, %read_token_b = xw.load %a
      : (!xw.ptr<#xw.local>) -> (!xw.simd<i32, 16>, !xw.mem.token)
  return
}

// CHECK-LABEL: func.func @independent_and_reads
// CHECK: %[[STORE_A:.*]] = xw.store
// CHECK-NEXT: %[[STORE_B:.*]] = xw.store
// CHECK-NOT: after %[[STORE_A]]
// CHECK: %{{.*}}, %[[READ_A:.*]] = xw.load {{.*}} after %[[STORE_A]]
// CHECK: %{{.*}}, %[[READ_B:.*]] = xw.load {{.*}} after %[[STORE_A]]
// CHECK-NOT: after %[[READ_A]]

// -----

func.func @alias_hazards_and_barrier(%value: !xw.simd<i32, 16>) attributes {
    xw.simd_width = 16 : i32} {
  %a = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %b = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %store_a = xw.store %value -> %a
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.local>) -> !xw.mem.token
  %store_b = xw.store %value -> %b
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.local>) -> !xw.mem.token
  %read_a, %read_a_token = xw.load %a
      : (!xw.ptr<#xw.local>) -> (!xw.simd<i32, 16>, !xw.mem.token)
  %barrier = "xw.barrier"() : () -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @alias_hazards_and_barrier
// CHECK: %[[STORE_A:.*]] = xw.store
// CHECK: %[[STORE_B:.*]] = xw.store
// CHECK: xw.load {{.*}} after %[[STORE_A]]
// CHECK: %[[JOIN:.*]] = xw.join {{.*}}%[[STORE_A]]{{.*}}%[[STORE_B]]
// CHECK: xw.barrier %[[JOIN]]
