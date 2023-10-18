// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by mlir-opt (without any options.)
module {
  func.func @getPartitions(%A: tensor<?x?xf32>) -> tensor<?xindex> {
    %partition_plan = partition.get_partition %A: tensor<?x?xf32> -> tensor<?xindex>
    %c0 = arith.constant 0 : index
    %c2_i64 = arith.constant 2 : i64
    %n0 = tensor.dim %partition_plan, %c0 : tensor<?xindex>
    %n = arith.index_cast %n0: index to i64
    // dimSize is 2, tensor A is 2d and hence every point has 2 co-ordinates.
    %num_points = arith.divui %n, %c2_i64: i64
    %num_points_index = arith.index_cast %num_points: i64 to index
    // might want to add assert saying %n is divisible by 2
    %partition_plan_memref = memref.alloc(%n0): memref<?xindex>
    // hopefully these two memrefs (one created by bufferization and one allocated here are co-alesced by compiler)
    // as it's anyway read-only
    memref.tensor_store %partition_plan, %partition_plan_memref: memref<?xindex>
    scf.forall(%i) in (%num_points_index) {
      // not sure explicit index computation is needed %i_idx = arith.mul
      %v4 = vector.load %partition_plan_memref[%i]: memref<?xindex>, vector<2xindex>
      vector.print %v4 : vector<2xindex>
    }
    return %partition_plan : tensor<?xindex>
  }
}
