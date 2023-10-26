#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module{
func.func @local_relu(%argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c = arith.constant 0.0 : f32
    %0 = linalg.generic  #relu_memory_access_map 
      outs(%argx: tensor<?x?xf32>) {
        ^bb(%x: f32):
          %1 = arith.maxf %x, %c : f32
          linalg.yield %1 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

func.func @getPartitions(%A: tensor<?x?xf32, #partEncoding>) -> tensor<?x?xf32, #partEncoding> {
    %partition_plan = part_tensor.get_partitions %A:  tensor<?x?xf32, #partEncoding> -> tensor<?xindex>
    %c0 = arith.constant 0 : index
    %c4_index = arith.constant 4 : index
    %c2_i64 = arith.constant 2 : i64
    %num_points_index = part_tensor.get_num_partitions %A : tensor<?x?xf32, #partEncoding> -> index
    %A_out = scf.forall(%i)= (0) to (%num_points_index) step (%c4_index) shared_outs(%o1 = %A) -> (tensor<?x?xf32, #partEncoding>) {
      %part_spec = tensor.extract_slice %partition_plan[%i] [4] [1] : tensor<?xindex> to tensor<4xindex>
      %A_slice = part_tensor.get_slice %A, %part_spec : tensor<?x?xf32, #partEncoding>, tensor<4xindex> -> tensor<?x?xf32>
      %relu_out = call @local_relu(%A_slice) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      scf.forall.in_parallel {
        part_tensor.set_slice %relu_out, %o1, %part_spec : tensor<?x?xf32>, tensor<?x?xf32, #partEncoding>, tensor<4xindex>
      }
    }
    return %partition_plan : tensor<?xindex>
  }
}
