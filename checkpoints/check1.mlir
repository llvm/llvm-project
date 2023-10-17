module {
  func.func @my_function(%arg0 : i32) {
	%0 = partition.fromPtr %arg0 : i32 to i32*
    return
  }
}
