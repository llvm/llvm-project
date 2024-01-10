// This is prettified example, this won't compile via mlir-opt.
#P = #part_tensor.encoding<{
  type="partitioned"
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module{
func.func @relu_test(%A: tensor<?x?xf32, #P>) -> tensor<?x?xf32, #P> {
    %part_list = part_tensor.get_partitions %A:  tensor<?x?xf32, #P> -> tensor<?xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c4_index = arith.constant 4 : index
    %num_parts = part_tensor.get_num_partitions %A : tensor<?x?xf32, #P> -> index
    %A_out = scf.for %i = %c0 to %num_parts step %c1 iter_args(%o1 = %A) -> (tensor<?x?xf32, #partEncoding>) {
      %i_offset = arith.muli %i, %c1: index
      %p = tensor.extract_slice %partition_plan[%i_offset] [4] [1] : tensor<?xindex> to tensor<4xindex>
      %s = part_tensor.get_slice %A, %part_spec : tensor<?x?xf32, #P>, tensor<4xindex> -> tensor<?x?xf32>
      %t = linalg.generic  #relu_memory_access_map
        outs(%s: tensor<?x?xf32>) {
          ^bb(%x: f32):
            %1 = arith.maxf %x, %c0_f32 : f32
            linalg.yield %1 : f32
      } -> tensor<?x?xf32>
      part_tensor.set_slice %t, %o1, %part_spec : tensor<?x?xf32>, tensor<?x?xf32, #P>, tensor<4xindex>
      scf.yield %o1 : tensor<?x?xf32, #P>
    }
    return %A_out: tensor<?x?xf32, #P>
  }
}
