// Inter compilation pipeline library.

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @match_func(
      %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["func.func"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @inter_lower_to_machine(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = transform.apply_registered_pass "inter-import-llvm" to %root
        : (!transform.any_op) -> !transform.any_op
    %r0a = transform.apply_registered_pass "inter-discover-cache-controls" to %r0
        : (!transform.any_op) -> !transform.any_op
    %r0b = transform.apply_registered_pass "inter-canonicalize-block2d-abi" to %r0a
        : (!transform.any_op) -> !transform.any_op
    %r0c = transform.apply_registered_pass "inter-canonicalize-dpas-builtin" to %r0b
        : (!transform.any_op) -> !transform.any_op
    %r1 = transform.apply_registered_pass "lift-cf-to-scf" to %r0c
        : (!transform.any_op) -> !transform.any_op
    %r1a = transform.apply_registered_pass "canonicalize" to %r1
        : (!transform.any_op) -> !transform.any_op
    %r1b = transform.apply_registered_pass "inter-prepare-counted-loops" to %r1a
        : (!transform.any_op) -> !transform.any_op
    %r2 = transform.apply_registered_pass "inter-verify-structured" to %r1b
        : (!transform.any_op) -> !transform.any_op
    %r3 = transform.apply_registered_pass "inter-convert-llvm-to-xw" to %r2
        : (!transform.any_op) -> !transform.any_op
    %r4 = transform.apply_registered_pass "inter-refine-distribution" to %r3
        : (!transform.any_op) -> !transform.any_op
    %r4a = transform.apply_registered_pass "canonicalize" to %r4
        : (!transform.any_op) -> !transform.any_op
    %r4b = transform.apply_registered_pass "inter-narrow-integer-ranges" to %r4a
        : (!transform.any_op) -> !transform.any_op
    %r4c = transform.apply_registered_pass "canonicalize" to %r4b
        : (!transform.any_op) -> !transform.any_op
    %r4d = transform.apply_registered_pass "inter-expand-arithmetic" to %r4c
        : (!transform.any_op) -> !transform.any_op
    %r5 = transform.apply_registered_pass "canonicalize" to %r4d
        : (!transform.any_op) -> !transform.any_op
    %r6 = transform.apply_registered_pass "cse" to %r5
        : (!transform.any_op) -> !transform.any_op
    %r7 = transform.apply_registered_pass "inter-infer-memory-tokens" to %r6
        : (!transform.any_op) -> !transform.any_op
    %r8 = transform.apply_registered_pass "inter-select-to-machine" to %r7
        : (!transform.any_op) -> !transform.any_op
    %funcs0 = transform.collect_matching @match_func in %r8
        : (!transform.any_op) -> !transform.any_op
    %funcs1 = transform.apply_registered_pass "inter-coalesce-tuples" to %funcs0
        : (!transform.any_op) -> !transform.any_op
    %funcs2 = transform.apply_registered_pass "inter-reuse-block2d-payloads" to %funcs1
        : (!transform.any_op) -> !transform.any_op
    %r9 = transform.apply_registered_pass "loop-invariant-code-motion" to %r8
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r9 : !transform.any_op
  }

  transform.named_sequence private @inter_prepare_regalloc(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %funcs0 = transform.collect_matching @match_func in %root
        : (!transform.any_op) -> !transform.any_op
    %funcs1 = transform.apply_registered_pass "inter-prepare-regalloc" to %funcs0
        : (!transform.any_op) -> !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence private @inter_allocate_registers(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = xemachine.transform.regalloc_arf_build_state from %root
        : (!transform.any_op) -> !transform.any_op
    %r1 = xemachine.transform.regalloc_arf_linear_scan from %r0
        : (!transform.any_op) -> !transform.any_op
    %r2 = xemachine.transform.regalloc_loop from %r1
        body = @inter_regalloc_iteration
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r2 : !transform.any_op
  }

  transform.named_sequence @inter_regalloc(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %r0 = transform.include @inter_prepare_regalloc failures(propagate) (%root)
        : (!transform.any_op) -> !transform.any_op
    %r1 = transform.include @inter_allocate_registers failures(propagate) (%r0)
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r1 : !transform.any_op
  }

  transform.named_sequence private @inter_regalloc_iteration(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = xemachine.transform.regalloc_build_state from %root
        : (!transform.any_op) -> !transform.any_op
    %r1 = xemachine.transform.regalloc_linear_scan from %r0
        : (!transform.any_op) -> !transform.any_op
    %r2 = xemachine.transform.regalloc_remat_relief from %r1
        : (!transform.any_op) -> !transform.any_op
    %r3 = xemachine.transform.regalloc_scratch_relief from %r2
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r3 : !transform.any_op
  }

  transform.named_sequence @inter_backend_no_sync(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = transform.include @inter_lower_to_machine failures(propagate) (%root)
        : (!transform.any_op) -> !transform.any_op
    %r1 = transform.include @inter_prepare_regalloc failures(propagate) (%r0)
        : (!transform.any_op) -> !transform.any_op
    %funcs0 = transform.collect_matching @match_func in %r1
        : (!transform.any_op) -> !transform.any_op
    %funcs1 = transform.apply_registered_pass "inter-machine-schedule" to %funcs0
        : (!transform.any_op) -> !transform.any_op
    %r2 = transform.include @inter_allocate_registers failures(propagate) (%r1)
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r2 : !transform.any_op
  }

  transform.named_sequence @inter_backend(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = transform.include @inter_backend_no_sync failures(propagate) (%root)
        : (!transform.any_op) -> !transform.any_op
    %funcs0 = transform.collect_matching @match_func in %r0
        : (!transform.any_op) -> !transform.any_op
    %funcs1 = transform.apply_registered_pass "inter-insert-sync" to %funcs0
        : (!transform.any_op) -> !transform.any_op
    %funcs2 = transform.apply_registered_pass "inter-resource-info" to %funcs1
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %r0 = transform.include @inter_backend failures(propagate) (%root)
        : (!transform.any_op) -> !transform.any_op
    transform.yield %r0 : !transform.any_op
  }
}
