module {
  func.func @cast_memref(%src: memref<4xi32>) -> memref<4xf32> {
    %src_cast = partition.fromPtr  %src : memref<4xi32> to memref<4xf32>
    return %src_cast : memref<4xf32>
  }
  func.func @getPartitions(%A: tensor<?x?xf32>) -> tensor<?xf32>{
	  %getAll = partition.get_partition %A: tensor<?x?xf32> -> tensor<?xf32>
	  return %getAll : tensor<?xf32>
	  }
}
