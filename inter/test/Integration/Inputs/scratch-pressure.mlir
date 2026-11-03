// Test-only Transform sequence that constrains the allocatable GRF suffix.

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @match_func(
      %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["func.func"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @set_scratch_pressure(
      %root: !transform.any_op {transform.readonly}) {
    %funcs = transform.collect_matching @match_func in %root
        : (!transform.any_op) -> !transform.any_op
    %reserved = transform.param.constant 113 : i32 -> !transform.param<i32>
    transform.annotate %funcs "xemachine.reserved_grf_count" = %reserved
        : !transform.any_op, !transform.param<i32>
    transform.yield
  }
}
