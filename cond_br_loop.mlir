llvm.func @loop_cond_br(%n: i32) -> i32 {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one  = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%zero, %zero : i32, i32)   // i=0, sum=0

^header(%i: i32, %sum: i32):
  %done = llvm.icmp "sge" %i, %n : i32       // i >= n ?
  llvm.cond_br %done, ^exit, ^body

^body:
  %sum2 = llvm.add %sum, %i : i32
  %i2   = llvm.add %i,   %one : i32
  llvm.br ^header(%i2, %sum2 : i32, i32)     // back-edge

^exit:
  llvm.return %sum : i32
}