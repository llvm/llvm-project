#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module{
func.func @getPartitions(%A: tensor<?x?xf32, #partEncoding>) -> tensor<?x?xf32, #partEncoding> {
    %partition_plan = part_tensor.get_partitions %A:  tensor<?x?xf32, #partEncoding> -> tensor<?xindex>
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %num_points_index = part_tensor.get_num_partitions %A : tensor<?x?xf32, #partEncoding> -> index
    %A_out = scf.for %i = %c0 to %num_points_index step %c1 iter_args(%o1 = %A) -> (tensor<?x?xf32, #partEncoding>) {
      %i_offset = arith.muli %i, %c4_index: index
      %part_spec = tensor.extract_slice %partition_plan[%i_offset] [4] [1] : tensor<?xindex> to tensor<4xindex>
      %A_slice = part_tensor.get_slice %A, %part_spec : tensor<?x?xf32, #partEncoding>, tensor<4xindex> -> tensor<?x?xf32>
      %relu_out = linalg.generic  #relu_memory_access_map
        outs(%A_slice: tensor<?x?xf32>) {
          ^bb(%x: f32):
            %1 = arith.maxf %x, %c0_f32 : f32
            linalg.yield %1 : f32
      } -> tensor<?x?xf32>
      part_tensor.set_slice %relu_out, %o1, %part_spec : tensor<?x?xf32>, tensor<?x?xf32, #partEncoding>, tensor<4xindex>
      scf.yield %o1 : tensor<?x?xf32, #partEncoding>
    }
    return %A_out: tensor<?x?xf32, #partEncoding>
  }
}
