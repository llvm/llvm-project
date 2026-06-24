module {
  func.func @add_dynamic(%arg0: memref<?x?bf16>, %arg1: memref<?x?bf16>, %arg2: memref<?x?bf16>) {
    linalg.add ins(%arg0, %arg1 : memref<?x?bf16>, memref<?x?bf16>) outs(%arg2 : memref<?x?bf16>)
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.add"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 16, 4] : !transform.any_op
      transform.yield 
    }
  }
}
